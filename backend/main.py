import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
_backend_env = Path(__file__).resolve().parent / ".env"
if _backend_env.exists():
    load_dotenv(dotenv_path=_backend_env, override=False)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import db, fs
from app.routers import course, lesson, pdf, progress, chunks, autotest, topics, objectives

app = FastAPI(title="AI Learning Platform", version="2.2.0")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf.router,         prefix="/api")
app.include_router(course.router,      prefix="/api")
app.include_router(lesson.router,      prefix="/api")
app.include_router(progress.router,    prefix="/api")
app.include_router(chunks.router,      prefix="/api")
app.include_router(autotest.router,    prefix="/api")
app.include_router(topics.router,      prefix="/api")
app.include_router(objectives.router,  prefix="/api")

@app.get("/api/health")
async def health_check():
    provider = (os.getenv("AI_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "heuristic"))
    return {"status": "healthy", "ai_provider": provider, "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
