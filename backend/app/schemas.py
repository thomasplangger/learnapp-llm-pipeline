# app/schemas.py
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Literal, Union, Dict, Any
from datetime import datetime
from app.models import Course, Lesson as LessonModel
from pydantic import BaseModel, Extra
from typing import List, Optional, Dict, Any
from datetime import datetime

# ─── Progress & Outlines ──────────────────────────────────────────────────
class ProgressIn(BaseModel):
    user_id: str
    lesson_id: str
    course_id: str
    completed: bool = False
    score: Optional[int]
    time_spent: int = 0
    completed_at: Optional[datetime]
    answers: Optional[Dict[int, Any]]

class OutlineLesson(BaseModel):
    id: str
    title: str
    summary: str
    section_title: Optional[str]

class OutlineResponse(BaseModel):
    course_id: str
    course_title: str
    course_description: str
    lessons: List[OutlineLesson]

# ─── Question Types & Lesson ───────────────────────────────────────────────
class QuestionBase(BaseModel):
    id: str
    question: str
    points: int

class MCQ(QuestionBase):
    type: Literal["mcq"]
    options: List[str]
    correct_answer: str

class TextQ(QuestionBase):
    type: Literal["text"]

Question = Union[MCQ, TextQ]

class LessonSchema(BaseModel):
    id: str
    course_id: str
    title: str
    summary: str
    content: str
    key_points: List[Any]
    questions: List[Dict[str, Any]]
    code_challenges: List[Dict[str, Any]]
    task_type: str
    difficulty: str = "beginner"
    estimated_time: int = 10
    section_title: Optional[str] = None

    model_config = dict(extra=Extra.ignore)

class PDFDetailRequest(BaseModel):
    course_id: str
    lesson_id: str
    lesson_title: str
    section_title: Optional[str]
    num_questions: int = 5

class CourseResponse(BaseModel):
    course: Course
    lessons: List[LessonModel]

class EvaluateRequest(BaseModel):
    lesson_id: str
    question_index: int
    user_answer: str

class EvaluateResponse(BaseModel):
    score: int
    explanation: str

class Chunk(BaseModel):
    id: str
    pdf_id: Optional[str] = None
    course_id: Optional[str] = None
    index: int
    start: int
    end: int
    text: str
    token_estimate: int
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime

class ChunkPlanRequest(BaseModel):
    target_tokens: int = 900
    min_tokens: int = 400
    max_tokens: int = 1200
    overwrite: bool = True
    dry_run: bool = False

