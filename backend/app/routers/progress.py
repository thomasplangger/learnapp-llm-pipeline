# app/routers/progress.py
from fastapi import APIRouter, HTTPException
from app.db import progress_collection
from app.schemas import ProgressIn

router = APIRouter(tags=["progress"])

@router.get("/progress/{user_id}")
async def get_progress(user_id: str):
    docs = await progress_collection.find({"user_id": user_id}).to_list(length=1000)
    for d in docs:
        d.pop("_id", None)
    return docs

@router.post("/progress")
async def update_progress(p: ProgressIn):
    if p.answers:
        p.answers = { str(k): v for k, v in p.answers.items() }

    filter = {"user_id": p.user_id, "lesson_id": p.lesson_id}
    doc = p.dict()
    await progress_collection.replace_one(filter, doc, upsert=True)
    return {"status": "ok"}
