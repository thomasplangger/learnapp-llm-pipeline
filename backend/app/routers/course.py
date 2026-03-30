from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from bson import ObjectId
from app.db import courses_collection, lessons_collection, pdfs_metadata_coll, chunks_collection
from app.models import Course, Lesson
from app.schemas import CourseResponse
from datetime import datetime
import uuid
import math
import asyncio
from app.utils.openai_client import chat_completion, strip_markdown_json

router = APIRouter(tags=["courses"])

@router.get("/courses", response_model=List[Course])
async def list_courses():
    docs = await courses_collection.find().to_list(length=100)
    for d in docs:
        d.pop("_id", None)
    return [Course(**d) for d in docs]

@router.get("/course/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lessons = await lessons_collection.find({
        "course_id": course_id,
        "id": {"$in": course["lessons"]}
    }).to_list(length=100)
    for l in lessons:
        l.pop("_id", None)
    return CourseResponse(
        course=Course(**{**course, **{"lessons": course["lessons"]}}),
        lessons=[Lesson(**l) for l in lessons]
    )

@router.get("/course/{course_id}/pdfs")
async def get_course_pdfs(course_id: str):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    pdf_ids = course.get("pdf_ids", [])
    out = []
    for pid in pdf_ids:
        try:
            oid = ObjectId(pid)
        except Exception:
            continue
        meta = await pdfs_metadata_coll.find_one({"_id": oid})
        if meta:
            out.append({
                "pdf_id": pid,
                "filename": meta.get("filename", "document.pdf"),
                "page_count": meta.get("page_count", 0),
                "word_count": meta.get("word_count", 0),
                "uploaded_at": meta.get("uploaded_at"),
            })
    return out

@router.delete("/course/{course_id}", status_code=204)
async def delete_course(course_id: str):
    await lessons_collection.delete_many({"course_id": course_id})
    res = await courses_collection.delete_one({"id": course_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Course not found")
    return


async def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


async def _generate_lesson_with_chunk(lesson_doc: Dict[str, Any]) -> Dict[str, Any]:
    lesson_id = lesson_doc.get("id")
    course_id = lesson_doc.get("course_id")
    title = lesson_doc.get("title") or "Lesson"
    section_title = lesson_doc.get("section_title") or "General"
    source = lesson_doc.get("source") or {}
    chunk_index = int(source.get("chunk_index", -1))
    part_index = int(source.get("part_index", 1))
    part_count = int(source.get("part_count", 1))

    row = await chunks_collection.find_one({"course_id": course_id, "index": chunk_index})
    if not row:
        raise HTTPException(status_code=404, detail=f"Chunk not found for index {chunk_index}")
    chunk_text = row.get("text") or ""

    MAX_CTX_CHARS = 12000
    ctx = chunk_text[:MAX_CTX_CHARS]

    prompt = f"""
You are an expert educational content creator.
Lesson title: {title}
Section: {section_title}

This lesson is derived from a specific source chunk of course material.
Part info: this is part {part_index} of {part_count} for this chunk. Avoid overlapping too much with other parts; focus on coherent subthemes.

Source Chunk Text (use as the primary grounding):
"""
    prompt = (prompt + ctx).strip()

    schema = f"""

Return valid JSON ONLY, in this exact schema:
{{
  "id": "{lesson_id}",
  "title": "{title}",
  "summary": "...",
  "content": "2-3 short paragraphs grounded in the source",
  "key_points": ["...", "...", "..."],
  "questions": [
    {{ "MCQ": {{"id": "q1", "question": "...", "options": ["A) ...","B) ...","C) ...","D) ..."], "correct_answer": "A", "explanation": "...", "points": 1}}}},
    {{ "TextQ": {{"id": "q2", "question": "...", "correct_answer": "...", "explanation": "...", "points": 2}}}}
  ],
  "code_challenges": [
    {{"title": "...", "description": "...", "starter_code": "...", "solution": "...", "difficulty": "easy"}}
  ]
}}

Rules:
- Ensure the TOTAL of all question "points" equals exactly 10.
- Ground all content in the provided Source Chunk Text.
- No markdown fences; JSON only.
""".strip()

    raw = await asyncio.to_thread(
        lambda: chat_completion(
            [
                {"role": "system", "content": "You are an expert educational content creator."},
                {"role": "user",   "content": prompt + "\n\n" + schema},
            ],
            max_tokens=1200
        )
    )
    clean = strip_markdown_json(raw)
    import json as _json
    try:
        data = _json.loads(clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from AI: {e}")

    flat_qs = []
    for q in data.get("questions", []):
        if "MCQ" in q:
            item = q["MCQ"]; item["type"] = "MCQ"; flat_qs.append(item)
        elif "TextQ" in q:
            item = q["TextQ"]; item["type"] = "TextQ"; flat_qs.append(item)
        else:
            q["type"] = q.get("type", "MCQ"); flat_qs.append(q)
    data["questions"] = flat_qs

    from app.schemas import LessonSchema
    lesson = LessonSchema(
        **data,
        course_id=course_id,
        task_type="qa",
        difficulty="beginner",
        section_title=section_title,
    )
    await lessons_collection.replace_one(
        {"id": lesson_id},
        {**lesson.dict(), "generation_status": "ready", "source": source},
        upsert=True,
    )
    return lesson.dict()


async def _generate_lo_chain(course_id: str, lo_lessons: List[Dict[str, Any]]):
    """Generate lessons sequentially for one LO: when one finishes, trigger the next."""
    for ldoc in lo_lessons:
        try:
            await lessons_collection.update_one({"id": ldoc["id"]}, {"$set": {"generation_status": "in_progress"}})
            await _generate_lesson_with_chunk(ldoc)
        except Exception:
            await lessons_collection.update_one({"id": ldoc.get("id")}, {"$set": {"generation_status": "error"}})
            continue


@router.post("/course/{course_id}/create-lessons-from-los")
async def create_lessons_from_los(course_id: str):
    """Generate lessons based on persisted learning objectives on the course.
    New behavior:
    - For each LO's member chunks, create ~1 lesson per 300 tokens (ceil), min 1.
    - Attach source metadata (chunk_index, part_index, part_count).
    - Immediately start background generation of the first lesson per LO, then chain the rest sequentially.
    - If no chunk membership is available, fall back to 1 lesson per LO objective bullet.
    """
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    los = list(course.get("learning_objectives") or [])
    if not los:
        raise HTTPException(status_code=422, detail="No learning objectives on course")

    await lessons_collection.delete_many({"course_id": course_id})

    rows = await chunks_collection.find({"course_id": course_id}).sort("index", 1).to_list(length=5000)
    by_idx = {int(r.get("index")): r for r in rows}

    new_ids: List[str] = []
    stubs: List[Dict[str, Any]] = []
    lo_to_lessons: List[List[Dict[str, Any]]] = []

    for lo in los:
        section_title = (lo.get("title") or "Learning Objective").strip()
        summary = (lo.get("summary") or "").strip()
        members = [int(i) for i in (lo.get("chunk_indices") or [])]

        lo_lessons: List[Dict[str, Any]] = []

        if members:
            for ci in members:
                row = by_idx.get(ci)
                if not row:
                    continue
                tok = int(row.get("token_estimate") or await _approx_tokens(row.get("text") or ""))
                parts = max(1, math.ceil(tok / 300))
                for j in range(parts):
                    lid = str(uuid.uuid4())
                    objectives = lo.get("objectives") or []
                    base_title = objectives[j] if j < len(objectives) else (
                        (((row.get("meta") or {}).get("enriched") or {}).get("title")
                         or (row.get("meta") or {}).get("header")
                         or section_title)
                    )
                    title = (str(base_title).strip() or section_title)[:140]
                    if parts > 1:
                        title = f"{title} (Part {j+1}/{parts})"
                    lesson_doc = {
                        "id": lid,
                        "course_id": course_id,
                        "title": title,
                        "summary": summary,
                        "content": "",
                        "key_points": [],
                        "questions": [],
                        "code_challenges": [],
                        "task_type": "qa",
                        "difficulty": "beginner",
                        "estimated_time": 0,
                        "section_title": section_title,
                        "generation_status": "pending",
                        "source": {"type": "chunk", "chunk_index": ci, "part_index": j+1, "part_count": parts},
                    }
                    await lessons_collection.insert_one(lesson_doc)
                    new_ids.append(lid)
                    stubs.append({"id": lid, "title": title, "summary": summary, "section_title": section_title})
                    lo_lessons.append(lesson_doc)
        else:
            objectives = lo.get("objectives") or []
            lesson_titles = objectives if objectives else [section_title]
            for obj_title in lesson_titles:
                lid = str(uuid.uuid4())
                title = (str(obj_title or section_title).strip()[:120] or section_title)
                lesson_doc = {
                    "id": lid,
                    "course_id": course_id,
                    "title": title,
                    "summary": summary,
                    "content": "",
                    "key_points": [],
                    "questions": [],
                    "code_challenges": [],
                    "task_type": "qa",
                    "difficulty": "beginner",
                    "estimated_time": 0,
                    "section_title": section_title,
                    "generation_status": "pending",
                }
                await lessons_collection.insert_one(lesson_doc)
                new_ids.append(lid)
                stubs.append({"id": lid, "title": title, "summary": summary, "section_title": section_title})
                lo_lessons.append(lesson_doc)

        lo_to_lessons.append(lo_lessons)

    await courses_collection.update_one(
        {"id": course_id},
        {"$set": {"lessons": new_ids, "updated_at": datetime.utcnow()}},
    )
    for lo_lessons in lo_to_lessons:
        if not lo_lessons:
            continue
        asyncio.create_task(_generate_lo_chain(course_id, lo_lessons))

    return {"created": len(new_ids), "lessons": stubs, "background_generation": True}
