from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Course(BaseModel):
    id: str
    title: str
    description: str
    lessons: List[str]
    source_type: str
    language: Optional[str] = None
    level: str = "beginner"
    created_at: datetime
    created_by: str

class Lesson(BaseModel):
    id: str
    course_id: str
    title: str
    summary: str
    content: str
    key_points: List[str]
    questions: List[Dict[str, Any]]
    code_challenges: List[Dict[str, Any]]
    task_type: str
    difficulty: str = "beginner"
    estimated_time: int = 10
    section_title: Optional[str] = None

class Progress(BaseModel):
    id: Optional[str] = None
    user_id: str
    lesson_id: str
    course_id: str
    completed: bool = False
    score: Optional[int] = None
    time_spent: int = 0
    completed_at: Optional[datetime] = None
    answers: Optional[Dict[str, Any]] = None
