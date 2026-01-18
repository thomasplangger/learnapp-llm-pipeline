# app/routers/lesson.py
import json
import uuid
from fastapi import APIRouter, HTTPException
from app.db import lessons_collection
from app.schemas import PDFDetailRequest, LessonSchema, EvaluateRequest, EvaluateResponse
from app.utils.openai_client import chat_completion, strip_markdown_json, evaluate_text_answer

router = APIRouter(tags=["lesson"])

@router.post("/generate-lesson-detail", response_model=LessonSchema)
async def generate_lesson_detail(req: PDFDetailRequest):
    TOTAL_POINTS = 10

    prompt = f"""
You are an expert educational content creator.
Lesson title: {req.lesson_title}
Section: {req.section_title or 'General'}

Generate a list of questions whose point-values sum to exactly {TOTAL_POINTS}.
- MCQ questions are worth 1 point each and must be formatted as:
  {{ "MCQ": {{
      "id": "q1",
      "question": "...",
      "options": ["A) …", "B) …", "C) …", "D) …"],
      "correct_answer": "A",
      "explanation": "...",
      "points": 1
    }} }}
- TextQ questions are worth 2 points each and must be formatted as:
  {{ "TextQ": {{
      "id": "q2",
      "question": "...",
      "correct_answer": "...",
      "explanation": "...",
      "points": 2
    }} }}

Keep adding MCQs or TextQs (choosing which makes sense) until the **sum of all "points" equals {TOTAL_POINTS}**.

Finally, wrap your questions array and a single code challenge into this JSON schema (no extra keys):
{{
  "id": "{req.lesson_id}",
  "title": "{req.lesson_title}",
  "summary": "...",
  "content": "2–3 paragraphs…",
  "key_points": ["…", "…", "…"],
  "questions": [ /* your MCQ/TextQ entries here */ ],
  "code_challenges": [
    {{
      "title": "…",
      "description": "…",
      "starter_code": "…",
      "solution": "…",
      "difficulty": "easy"
    }}
  ]
}}
Return **valid JSON only**—no markdown, no bullet lists outside the JSON.
""".strip()

    raw = chat_completion(
        [
            {"role":"system", "content":"You are an expert educational content creator."},
            {"role":"user",   "content":prompt}
        ],
        max_tokens=1200
    )
    clean = strip_markdown_json(raw)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON from AI: {e}")

    flat_qs = []
    for item in data.get("questions", []):
        if "MCQ" in item:
            q = item["MCQ"]
            q["type"] = "MCQ"
            flat_qs.append(q)
        elif "TextQ" in item:
            q = item["TextQ"]
            q["type"] = "TextQ"
            flat_qs.append(q)
        else:
            item.setdefault("type","MCQ")
            flat_qs.append(item)
    data["questions"] = flat_qs

    lesson = LessonSchema(
        **data,
        course_id=req.course_id,
        task_type="qa",
        difficulty="beginner",
        section_title=req.section_title
    )
    await lessons_collection.replace_one(
        {"id": req.lesson_id},
        lesson.dict(),
        upsert=True
    )
    return lesson


@router.get("/lesson/{lesson_id}", response_model=LessonSchema)
async def get_lesson(lesson_id: str):
    doc = await lessons_collection.find_one({"id": lesson_id})
    if not doc:
        raise HTTPException(404, "Lesson not found")
    doc.pop("_id", None)
    return LessonSchema(**doc)


@router.post("/evaluate-answer", response_model=EvaluateResponse)
async def evaluate_answer(req: EvaluateRequest):
    lesson = await lessons_collection.find_one({"id": req.lesson_id})
    if not lesson:
        raise HTTPException(404, "Lesson not found")

    questions = lesson.get("questions", [])
    if not (0 <= req.question_index < len(questions)):
        raise HTTPException(400, "Invalid question index")

    qobj = questions[req.question_index]
    question_text  = qobj.get("question", "")
    correct_answer = qobj.get("correct_answer", "")

    if qobj.get("type") == "TextQ":
        score, explanation = await evaluate_text_answer(
            question_text,
            correct_answer,
            req.user_answer
        )
    else:
        correct = req.user_answer.strip().upper() == correct_answer.strip().upper()
        score = 1 if correct else 0
        explanation = qobj.get("explanation", "")
    return EvaluateResponse(score=score, explanation=explanation)
