# LearnApp – LLM-Based Course Generation & Evaluation Platform

End-to-end AI-assisted learning platform that transforms raw educational PDFs into structured courses consisting of semantically coherent content chunks, learning objectives, and lessons.

Developed as part of my MSc thesis in Computer Science (TU Graz).

---

## Overview

LearnApp was built to reduce manual course-authoring effort by converting unstructured educational documents into usable learning units.

The system focuses on three main research and engineering components:

1. **LLM-Based Semantic Chunking**
2. **Learning Objective Generation & Grouping**
3. **Autotest & Evaluation Framework**

Together, these form a modular pipeline that combines applied NLP/LLM integration with full-stack system design.

---

## 1) LLM-Based Semantic Chunking

The core challenge addressed in this project is splitting long educational documents into semantically coherent segments.

Instead of naive fixed-length splitting, the system:
- Uses LLM-driven topic-awareness
- Preserves conceptual boundaries
- Produces meaningful units suitable for downstream processing

This component forms the backbone of the entire pipeline.

---

## 2) Learning Objective Generation & Grouping

After chunking, the system:

- Generates learning objectives from chunk content
- Groups chunks based on semantic similarity and topic signals
- Builds course structures around these objectives

The architecture supports:
- OpenAI-based generation
- Local heuristic fallback mode
- Manual refinement workflows via the UI

This allows both automated and guided course construction.

---

## 3) Autotest & Evaluation Framework

To move beyond “it looks good”, the system includes a dedicated evaluation layer.

Key features:
- Synthetic test corpus generation
- Automated benchmarking endpoints
- Repeatable test runs
- Validation of chunk boundaries and grouping behavior
- Support for experiment-style analysis

This enables systematic validation of LLM-driven outputs rather than purely qualitative inspection.

---

## System Architecture

PDF Upload  
→ Text Extraction  
→ Semantic Chunking  
→ Metadata Enrichment  
→ Learning Objective Generation & Grouping  
→ Lesson Construction  
→ MongoDB Storage  
→ React-Based Course Interface  

The backend exposes modular API routes that allow:
- Fully automated execution
- Stepwise debug workflows for inspecting intermediate pipeline states

---

## Core Features

- Multi-document course creation
- Topic-aware semantic segmentation
- Learning objective grouping from chunk representations
- Modular AI provider abstraction (`openai` vs `heuristic`)
- Lesson navigation UI
- Progress tracking
- Autotest & benchmark endpoints

---

## Tech Stack

### Backend
- FastAPI
- Pydantic
- MongoDB
- OpenAI API (optional)
- Modular provider abstraction

### Frontend
- React
- React Router
- Axios
- TailwindCSS

---

## Repository Structure

```
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
│   └── src/
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or hosted)

---

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Create `backend/.env`:

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=learnapp
AI_PROVIDER=heuristic
# OPENAI_API_KEY=sk-...    # required if using OpenAI provider
```

Health endpoint:
http://localhost:8001/api/health

---

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend:
http://localhost:3000

---

## Research Context

This repository was developed within the scope of my Master’s thesis:

**“Topic-Aware Semantic Segmentation and Learning Objective Induction from Educational Content using Large Language Models.”**

The project demonstrates:

- Applied LLM integration in production-style pipelines
- Topic-aware document segmentation
- Embedding-based grouping strategies
- Evaluation-driven development
- Full-stack system architecture (API + database + UI)

---

## License

MIT License
