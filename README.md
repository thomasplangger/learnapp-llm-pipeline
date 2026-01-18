# LearnApp Demo – Learning Objective Segmentation Platform

This repository contains a FastAPI backend and a React frontend for building AI-assisted learning courses from PDFs. The pipeline supports PDF ingestion, chunking, metadata enrichment, learning objective (LO) generation, and lesson creation.

---

## Repository Structure

```
.
├── backend/                          # FastAPI backend
│   ├── main.py                       # API entrypoint (uvicorn main:app)
│   ├── requirements.txt              # Backend dependencies
│   ├── generate_bench_texts_gui.py   # GUI tool to create test corpora
│   ├── app/
│   │   ├── routers/                  # API routes (pdf, course, chunks, objectives, lesson, progress, autotest)
│   │   ├── providers/                # AI providers (OpenAI, heuristic)
│   │   ├── services/                 # Chunking and pipeline services
│   │   ├── utils/                    # PDF processing, metadata helpers
│   │   ├── db.py                     # MongoDB setup
│   │   ├── models.py                 # Pydantic models
│   │   └── schemas.py                # API request/response schemas
│   ├── generated/                    # Not in repo: create via generate_bench_texts_gui.py
│   ├── testdata/                     # Not in repo: autotest outputs/caches
│   └── Tests/                        # Offline analysis scripts (A–E)
├── frontend/                         # React frontend
│   ├── package.json                  # Frontend dependencies
│   ├── public/
│   └── src/components/               # UI pages and components
└── requirements.txt                  # Root Python deps (if used)
```

---

## Setup Guide

### 1) Prerequisites

- Python 3.10+ and pip
- Node.js 18+ and npm or yarn
- MongoDB instance (local or cloud)
- OpenAI API key if you want LLM features

### 2) Environment Variables

Create two .env files:

Root .env (optional): for shared variables if you use custom scripts.

backend/.env (required):

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=learnapp
OPENAI_API_KEY=sk-...      # optional, enables LLM flows
AI_PROVIDER=openai         # or "heuristic" to stay fully local
CHUNK_DEBUG_DIR=debug      # optional, stores chunking traces
```

Frontend env (optional): frontend/.env with REACT_APP_BACKEND_URL if your backend is not on http://localhost:8001.

---

### 3) Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the API:

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Health check: http://localhost:8001/api/health

---

### 4) Frontend Setup

```bash
cd frontend
npm install        # or: yarn install
npm start          # or: yarn start
```

The dev server runs on http://localhost:3000.

---

## Typical Workflow

1) Upload PDFs in the UI.
2) Create a course, then choose Auto or Debug mode.
3) Auto mode runs:
   - Chunking
   - Chunk metadata
   - Learning objective generation (LLM default)
4) Debug mode lets you run each step manually.
5) Edit courses, LOs, and chunks in the Course Data page.

---

## API Routers (backend/app/routers)

| Router | Purpose |
| --- | --- |
| pdf.py | PDF upload, outline + lesson bootstrap |
| chunks.py | Chunking + metadata enrichment |
| lesson.py | Lesson CRUD and answer grading |
| objectives.py | LO grouping and editing |
| course.py | Course fetch and LO → lesson creation |
| progress.py | User progress tracking |
| autotest.py | Synthetic testing tools |

---

## Important Notes

- Autotest requires `backend/testdata` to exist. Create the folder before running any autotest runs.
- Autotest source data lives in `backend/generated`. Use `backend/generate_bench_texts_gui.py` to create it.
- LLM features require `OPENAI_API_KEY` and `AI_PROVIDER=openai`. For local-only behavior, set `AI_PROVIDER=heuristic`.
- If chunking outputs look off, enable `CHUNK_DEBUG_DIR` and inspect `backend/debug`.

---

## Author and Maintainer

Thomas Plangger BSc — Master Thesis @ TU Graz
Supervised by Priv.-Doz. Dipl.-Ing. Dr.techn. Martin Ebner
Scientific Advisors: Benedikt Brünner and Philipp Leitner

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
