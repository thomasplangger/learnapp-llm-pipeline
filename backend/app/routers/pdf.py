import io, json, uuid, re
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pdfminer.high_level import extract_text
from app.db import fs, pdfs_metadata_coll, courses_collection, lessons_collection, pdf_texts_coll
from app.schemas import OutlineResponse, OutlineLesson, PDFDetailRequest, LessonSchema
from app.utils.extract_pdf import extract_pdf_text_and_store
from app.utils.openai_client import chat_completion, strip_markdown_json
from app.utils.pdf_builder import build_pdf_from_text

router = APIRouter(tags=["pdf"])

class CreateCourseRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    num_lessons: int = 5
    pdf_ids: List[str]

async def _create_course_impl(req: CreateCourseRequest) -> OutlineResponse:
    if not req.pdf_ids:
        raise HTTPException(status_code=400, detail="pdf_ids must be non-empty")

    texts: List[str] = []
    for pid in req.pdf_ids:
        try:
            oid = ObjectId(pid)
            stream = await fs.open_download_stream(oid)
            content = await stream.read()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid pdf_id: {pid}")
        txt = extract_text(io.BytesIO(content)) or ""
        texts.append(txt)

    combined = "\n\n".join(texts)[:8000]

    prompt = f"""
You will outline a course based on multiple PDF sources (combined excerpt below).
Create a {req.num_lessons}-lesson outline, organized into chapters (learning objectives).
Return valid JSON exactly in this structure:

{{
  "course_title":"Main course title",
  "course_description":"Brief course description",
  "sections":[
    {{
      "section_title":"Learning Objective name",
      "lessons":[
        {{ "id":"l1","title":"Lesson title","summary":"1–2 sentence summary" }}
      ]
    }}
  ]
}}

Combined excerpt:
{combined}
""".strip()

    raw = chat_completion(
        [
            {"role": "system", "content": "You are an expert educational content creator."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=1500
    )
    clean = strip_markdown_json(raw)

    try:
        data = json.loads(clean)
    except Exception:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    stub_lessons = []
    lesson_ids = []
    for sec in data.get("sections", []):
        sec_title = sec.get("section_title")
        for stub in sec.get("lessons", []):
            lid = str(uuid.uuid4())
            lesson_ids.append(lid)
            stub_lessons.append({
                "id":            lid,
                "title":         stub["title"],
                "summary":       stub["summary"],
                "section_title": sec_title
            })
    if len(lesson_ids) > req.num_lessons:
        lesson_ids   = lesson_ids[:req.num_lessons]
        stub_lessons = stub_lessons[:req.num_lessons]

    final_title       = (req.title or data.get("course_title") or "New Course").strip()
    final_description = (req.description or data.get("course_description") or "").strip()

    course_id = str(uuid.uuid4())
    await courses_collection.insert_one({
        "id":           course_id,
        "title":        final_title,
        "description":  final_description,
        "lessons":      lesson_ids,
        "pdf_ids":      req.pdf_ids,
        "source_type":  "pdf",
        "created_at":   datetime.utcnow(),
        "created_by":   "default_user"
    })

    for stub in stub_lessons:
        await lessons_collection.insert_one({
            "id":              stub["id"],
            "course_id":       course_id,
            "title":           stub["title"],
            "summary":         stub["summary"],
            "content":         "",
            "key_points":      [],
            "questions":       [],
            "code_challenges": [],
            "task_type":       "qa",
            "difficulty":      "beginner",
            "estimated_time":  0,
            "section_title":   stub.get("section_title"),
        })

    return OutlineResponse(
        course_id=course_id,
        course_title=final_title,
        course_description=final_description,
        lessons=[OutlineLesson(**l) for l in stub_lessons]
    )

@router.post("/create-course", response_model=OutlineResponse)
async def create_course(req: CreateCourseRequest):
    return await _create_course_impl(req)

@router.post("/pdf/create-course", response_model=OutlineResponse)
async def create_course_alias(req: CreateCourseRequest):
    return await _create_course_impl(req)


@router.post("/create-course", response_model=OutlineResponse)
async def create_course(req: CreateCourseRequest):
    if not req.pdf_ids:
        raise HTTPException(status_code=400, detail="pdf_ids must be non-empty")

    texts: List[str] = []
    for pid in req.pdf_ids:
        try:
            oid = ObjectId(pid)
            stream = await fs.open_download_stream(oid)
            content = await stream.read()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid pdf_id: {pid}")
        txt = extract_text(io.BytesIO(content)) or ""
        texts.append(txt)

    combined_text = "\n\n".join(texts)
    snippet = combined_text[:8000]

    prompt = f"""
You will outline a course based on multiple PDF sources (combined excerpt below).
Create a {req.num_lessons}-lesson outline, organized into chapters (learning objectives).
Return valid JSON exactly in this structure:

{{
  "course_title":"Main course title",
  "course_description":"Brief course description",
  "sections":[
    {{
      "section_title":"Learning Objective name",
      "lessons":[
        {{ "id":"l1","title":"Lesson title","summary":"1–2 sentence summary" }}
      ]
    }}
  ]
}}

Combined excerpt:
{snippet}
""".strip()

    raw = chat_completion(
        [
            {"role": "system", "content": "You are an expert educational content creator."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=1500
    )
    clean = strip_markdown_json(raw)

    try:
        data = json.loads(clean)
    except Exception:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    stub_lessons = []
    lesson_ids = []
    for sec in data.get("sections", []):
        sec_title = sec.get("section_title")
        for stub in sec.get("lessons", []):
            lid = str(uuid.uuid4())
            lesson_ids.append(lid)
            stub_lessons.append({
                "id":            lid,
                "title":         stub["title"],
                "summary":       stub["summary"],
                "section_title": sec_title
            })

    if len(lesson_ids) > req.num_lessons:
        lesson_ids   = lesson_ids[:req.num_lessons]
        stub_lessons = stub_lessons[:req.num_lessons]

    final_title       = (req.title or data.get("course_title") or "New Course").strip()
    final_description = (req.description or data.get("course_description") or "").strip()

    course_id = str(uuid.uuid4())
    await courses_collection.insert_one({
        "id":           course_id,
        "title":        final_title,
        "description":  final_description,
        "lessons":      lesson_ids,
        "pdf_ids":      req.pdf_ids,
        "source_type":  "pdf",
        "created_at":   datetime.utcnow(),
        "created_by":   "default_user"
    })

    for stub in stub_lessons:
        await lessons_collection.insert_one({
            "id":              stub["id"],
            "course_id":       course_id,
            "title":           stub["title"],
            "summary":         stub["summary"],
            "content":         "",
            "key_points":      [],
            "questions":       [],
            "code_challenges": [],
            "task_type":       "qa",
            "difficulty":      "beginner",
            "estimated_time":  0,
            "section_title":   stub.get("section_title"),
        })

    return OutlineResponse(
        course_id=course_id,
        course_title=final_title,
        course_description=final_description,
        lessons=[OutlineLesson(**l) for l in stub_lessons]
    )

@router.get("/pdfs")
async def list_pdfs():
    items = await pdfs_metadata_coll.find().to_list(length=100)
    courses = await courses_collection.find(
        {},
        {"id": 1, "title": 1, "pdf_ids": 1}
    ).to_list(length=500)
    usage_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in courses:
        for pid in c.get("pdf_ids", []) or []:
            key = str(pid)
            usage_map.setdefault(key, []).append({
                "id": c.get("id"),
                "title": c.get("title"),
            })
    return [
        {
            "pdf_id":      str(item["_id"]),
            "filename":    item["filename"],
            "page_count":  item["page_count"],
            "word_count":  item["word_count"],
            "uploaded_at": item["uploaded_at"].isoformat(),
            "courses":     usage_map.get(str(item["_id"]), []),
        }
        for item in items
    ]

@router.get("/pdf/{pdf_id}/raw")
async def stream_pdf_raw(pdf_id: str):
    """Stream the original PDF bytes from GridFS for inline viewing.
    Returns `application/pdf` data so it can be embedded via an <iframe>.
    """
    try:
        oid = ObjectId(pdf_id)
        stream = await fs.open_download_stream(oid)
        data = await stream.read()
        if not data:
            raise HTTPException(status_code=404, detail="PDF is empty")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="PDF not found")
    return Response(content=data, media_type="application/pdf", headers={
        "Content-Disposition": f"inline; filename=\"{pdf_id}.pdf\""
    })

@router.put("/pdf/{pdf_id}")
async def rename_pdf(pdf_id: str, payload: dict):
    new_name = payload.get("filename")
    if not new_name:
        raise HTTPException(status_code=400, detail="filename required")
    try:
        oid = ObjectId(pdf_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")
    res = await pdfs_metadata_coll.update_one({"_id": oid}, {"$set": {"filename": new_name}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"pdf_id": pdf_id, "filename": new_name}

@router.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    return await extract_pdf_text_and_store(file)

@router.post("/generate-pdf-outline", response_model=OutlineResponse)
async def generate_pdf_outline(
    pdf_id: str,
    num_lessons: int = 5,
    title: str | None = Query(default=None),
    description: str | None = Query(default=None),
):
    try:
        oid = ObjectId(pdf_id)
        stream = await fs.open_download_stream(oid)
        content = await stream.read()
    except:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")

    text = extract_text(io.BytesIO(content))
    prompt = f"""
Create a {num_lessons}-lesson outline from this text, organized into chapters.
Return valid JSON:
{{
  "course_title":"Main course title",
  "course_description":"Brief course description",
  "sections":[
    {{
      "section_title":"Chapter name",
      "lessons":[
        {{ "id":"l1","title":"Lesson title","summary":"1–2 sentence summary" }}
      ]
    }}
  ]
}}
Text:
{text[:4000]}
""".strip()

    raw = chat_completion(
        [
            {"role":"system","content":"You are an expert educational content creator."},
            {"role":"user","content":prompt}
        ],
        max_tokens=1500
    )
    clean = strip_markdown_json(raw)

    try:
        data = json.loads(clean)
    except:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    stub_lessons = []
    lesson_ids = []
    for sec in data.get("sections", []):
        sec_title = sec.get("section_title")
        for stub in sec.get("lessons", []):
            lid = str(uuid.uuid4())
            lesson_ids.append(lid)
            stub_lessons.append({
                "id":            lid,
                "title":         stub["title"],
                "summary":       stub["summary"],
                "section_title": sec_title
            })

    if len(lesson_ids) > num_lessons:
        lesson_ids   = lesson_ids[:num_lessons]
        stub_lessons = stub_lessons[:num_lessons]

    final_title       = (title or data.get("course_title") or "New Course").strip()
    final_description = (description or data.get("course_description") or "").strip()

    outline_id = str(uuid.uuid4())
    await courses_collection.insert_one({
        "id":           outline_id,
        "title":        final_title,
        "description":  final_description,
        "lessons":      lesson_ids,
        "pdf_ids":      [pdf_id],
        "source_type":  "pdf",
        "created_at":   datetime.utcnow(),
        "created_by":   "default_user"
    })

    for stub in stub_lessons:
        await lessons_collection.insert_one({
            "id":              stub["id"],
            "course_id":       outline_id,
            "title":           stub["title"],
            "summary":         stub["summary"],
            "content":         "",
            "key_points":      [],
            "questions":       [],
            "code_challenges": [],
            "task_type":       "qa",
            "difficulty":      "beginner",
            "estimated_time":  0,
            "section_title":   stub.get("section_title"),
        })

    return OutlineResponse(
        course_id=outline_id,
        course_title=final_title,
        course_description=final_description,
        lessons=[OutlineLesson(**l) for l in stub_lessons]
    )

@router.post("/generate-pdf-lesson-detail", response_model=LessonSchema)
async def generate_pdf_lesson_detail(req: PDFDetailRequest):
    prompt = f"""
You are an expert educational content creator.
Lesson title: {req.lesson_title}
Section: {req.section_title or 'General'}

Generate a mixture of:
 - MCQ questions worth 1 point each
 - TextQ questions worth 2 points each

So that the **total sum of all question points is exactly 10**.
Do **not** restrict yourself to a fixed number of questions—choose as many as needed.

Return **valid JSON only**, matching this schema exactly:
{{
  "id":"{req.lesson_id}",
  "title":"{req.lesson_title}",
  "summary":"…",
  "content":"2–3 paragraphs…",
  "key_points":["…","…","…"],
  "questions":[
    {{ "MCQ":{{"id":"q1","question":"…","options":["A)…","B)…","C)…","D)…"],"correct_answer":"A","explanation":"…","points":1}}}},
    {{ "TextQ":{{"id":"q2","question":"…","correct_answer":"…","explanation":"…","points":2}}}}
    // repeat until total of all “points” fields = 10
  ],
  "code_challenges":[
    {{"title":"…","description":"…","starter_code":"…","solution":"…","difficulty":"easy"}}
  ]
}}
""".strip()

    raw = chat_completion(
      [
        {"role":"system","content":"You are an expert educational content creator."},
        {"role":"user","content":prompt}
      ],
      max_tokens=1200
    )
    clean = strip_markdown_json(raw)

    try:
      data = json.loads(clean)
    except json.JSONDecodeError as e:
      raise HTTPException(status_code=500, detail=f"Invalid JSON from AI: {e}")

    flat_qs = []
    for q in data.get("questions", []):
      if "MCQ" in q:
        item = q["MCQ"]
        item["type"] = "MCQ"
        flat_qs.append(item)
      elif "TextQ" in q:
        item = q["TextQ"]
        item["type"] = "TextQ"
        flat_qs.append(item)
      else:
        q["type"] = q.get("type", "MCQ")
        flat_qs.append(q)
    data["questions"] = flat_qs

    lesson = LessonSchema(
      **data,
      course_id=req.course_id,
      task_type="qa",
      difficulty="beginner",
      section_title=req.section_title,
    )
    await lessons_collection.replace_one(
      {"id": req.lesson_id},
      lesson.dict(),
      upsert=True
    )
    return lesson


@router.delete("/pdf/{pdf_id}", status_code=204)
async def delete_pdf(pdf_id: str):
    try:
        oid = ObjectId(pdf_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")

    res = await pdfs_metadata_coll.delete_one({"_id": oid})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="PDF not found")

    await fs.delete(oid)
    return

@router.get("/pdf/text/{pdf_id}")
async def get_pdf_text_api(pdf_id: str, limit: int = Query(default=0, ge=0)):
    try:
        oid = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")

    doc = await pdf_texts_coll.find_one({"_id": oid})
    if not doc:
        try:
            stream = await fs.open_download_stream(oid)
            raw = await stream.read()
            text = extract_text(io.BytesIO(raw)) or ""
        except Exception:
            raise HTTPException(status_code=404, detail="PDF not found or unreadable")
        doc = {"_id": oid, "text": text, "length": len(text), "created_at": datetime.utcnow()}
        await pdf_texts_coll.insert_one(doc)

    text = doc.get("text", "")
    if limit and limit > 0:
        text = text[:limit]

    meta = await pdfs_metadata_coll.find_one({"_id": oid}) or {}
    return {
        "pdf_id": pdf_id,
        "filename": meta.get("filename"),
        "length": len(doc.get("text", "")),
        "text": text
    }

@router.post("/pdf/backfill-text")
async def backfill_pdf_texts():
    metas = await pdfs_metadata_coll.find({}).to_list(length=10000)
    added, existing, errors = 0, 0, 0
    for m in metas:
        oid = m["_id"]
        if await pdf_texts_coll.find_one({"_id": oid}):
            existing += 1
            continue
        try:
            stream = await fs.open_download_stream(oid)
            raw = await stream.read()
            text = extract_text(io.BytesIO(raw)) or ""
            await pdf_texts_coll.insert_one({
                "_id": oid, "text": text, "length": len(text), "created_at": datetime.utcnow()
            })
            added += 1
        except Exception:
            errors += 1
            continue
    return {"added": added, "existing": existing, "errors": errors}

@router.get("/pdf/text/{pdf_id}")
async def get_pdf_text_api(pdf_id: str, limit: int = Query(default=0, ge=0)):
    try:
        oid = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")
    doc = await pdf_texts_coll.find_one({"_id": oid})
    if not doc:
        try:
            stream = await fs.open_download_stream(oid)
            raw = await stream.read()
            text = extract_text(io.BytesIO(raw)) or ""
        except Exception:
            raise HTTPException(status_code=404, detail="PDF not found or unreadable")
        doc = {"_id": oid, "text": text, "length": len(text), "created_at": datetime.utcnow()}
        await pdf_texts_coll.insert_one(doc)
    text = doc.get("text", "")
    if limit and limit > 0:
        text = text[:limit]
    meta = await pdfs_metadata_coll.find_one({"_id": oid}) or {}
    return {"pdf_id": pdf_id, "filename": meta.get("filename"), "length": len(doc.get("text","")), "text": text}

def _count_stats(text: str) -> dict:
    words = len(re.findall(r"\b\w+\b", text or ""))
    paragraphs = len([b for b in re.split(r"\n\s*\n+", text or "") if b.strip()])
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()])
    chars = len(text or "")
    tokens = max(1, chars // 4)
    return {"words": words, "paragraphs": paragraphs, "sentences": sentences, "chars": chars, "tokens": tokens}

@router.get("/pdf/stats/{pdf_id}")
async def get_pdf_stats(pdf_id: str):
    try:
        oid = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")
    doc = await pdf_texts_coll.find_one({"_id": oid})
    if not doc:
        try:
            stream = await fs.open_download_stream(oid)
            raw = await stream.read()
            text = extract_text(io.BytesIO(raw)) or ""
        except Exception:
            raise HTTPException(status_code=404, detail="PDF not found or unreadable")
        doc = {"_id": oid, "text": text}
        await pdf_texts_coll.insert_one(doc)
    stats = _count_stats(doc.get("text", ""))
    return {"pdf_id": pdf_id, **stats}

@router.post("/pdf/backfill-text")
async def backfill_pdf_texts():
    metas = await pdfs_metadata_coll.find({}).to_list(length=10000)
    added, existing, errors = 0, 0, 0
    for m in metas:
        oid = m["_id"]
        if await pdf_texts_coll.find_one({"_id": oid}):
            existing += 1; continue
        try:
            stream = await fs.open_download_stream(oid)
            raw = await stream.read()
            text = extract_text(io.BytesIO(raw)) or ""
            await pdf_texts_coll.insert_one({"_id": oid, "text": text, "length": len(text), "created_at": datetime.utcnow()})
            added += 1
        except Exception:
            errors += 1
    return {"added": added, "existing": existing, "errors": errors}
