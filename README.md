# LearnApp Demo

AI-assisted learning platform that transforms PDF content into structured courses with learning objectives, lessons, and progress tracking.

This project includes:
- A `FastAPI` backend for PDF processing, chunking, objective generation, and lesson workflows
- A `React` frontend for uploading documents, building courses, and navigating generated learning content
- Support for both `OpenAI` and local heuristic modes

## Why This Project

The core goal is to reduce manual course-authoring effort by turning raw source material into usable educational units.  
From an engineering perspective, the focus was on:
- Building an end-to-end content pipeline (upload -> parse -> chunk -> enrich -> structure)
- Designing APIs that support both automated and manual refinement workflows
- Keeping AI features optional via provider abstraction (`openai` vs `heuristic`)

## Core Features

- Multi-PDF course creation
- Learning objective grouping from chunk embeddings
- Lesson generation and lesson navigation UI
- Debug/stepwise workflow for inspecting pipeline stages
- Progress tracking and autotest utilities for evaluation experiments

## Tech Stack

- Backend: `FastAPI`, `Pydantic`, `MongoDB`
- Frontend: `React`, `React Router`, `Axios`, `TailwindCSS`
- AI: OpenAI API (optional) + local heuristic fallback

## Project Structure

```text
.
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── routers/
│   │   ├── providers/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── db.py
│   │   ├── models.py
│   │   └── schemas.py
│   └── Tests/
├── frontend/
│   ├── package.json
│   ├── public/
│   └── src/
└── README.md
```

## Local Setup

### Prerequisites

- Python `3.10+`
- Node.js `18+`
- MongoDB (local or hosted)

### 1) Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Create `backend/.env`:

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=learnapp
AI_PROVIDER=heuristic
# OPENAI_API_KEY=sk-...       # set when using OpenAI provider
```

Backend health endpoint:
- `http://localhost:8001/api/health`

### 2) Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at:
- `http://localhost:3000`

## API Modules

- `pdf.py`: PDF upload and course outline bootstrap
- `chunks.py`: chunking and metadata enrichment
- `objectives.py`: learning objective grouping and metadata
- `course.py`: course-level orchestration
- `lesson.py`: lesson CRUD and grading flows
- `progress.py`: learner progress tracking
- `autotest.py`: synthetic and benchmark-oriented test endpoints

## Portfolio Notes

This repository is part of my educational AI engineering work and research-driven development.  
It demonstrates full-stack development, applied NLP/LLM integration, data pipeline design, and API-first system architecture.

## License

MIT License. See `LICENSE`.
