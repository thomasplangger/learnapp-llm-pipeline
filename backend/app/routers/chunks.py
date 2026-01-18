# backend/app/routers/chunks.py
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.db import chunks_collection, courses_collection
from app.services.chunking import chunk_text_and_persist
from app.utils.pdf_to_markdown import (
    get_pdf_markdown,
    get_combined_markdown_and_offsets,
)
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(tags=["chunks"])

@router.get("/chunking/methods")
async def list_chunking_methods():
    return ["LLM [JSON]", "LLM [Raw]", "LLM [PDF]", "Context", "Other"]

@router.post("/pdf/{pdf_id}/chunk")
async def chunk_pdf(pdf_id: str, plan: dict):
    markdown = await get_pdf_markdown(pdf_id)
    if not markdown.strip():
        raise HTTPException(status_code=422, detail="No extractable text/markdown in PDF")

    try:
        return await chunk_text_and_persist(
            markdown,
            {"pdf_id": pdf_id, "course_id": None},
            {
                "method": plan.get("method", "LLM [JSON]"),
                "target_tokens": int(plan.get("target_tokens", 900)) if plan.get("method") == "Other" else None,
                "min_tokens": int(plan.get("min_tokens", 400)) if plan.get("method") == "Other" else None,
                "max_tokens": int(plan.get("max_tokens", 1200)) if plan.get("method") == "Other" else None,
                "desired_chunks": plan.get("desired_chunks"),
                "overwrite": bool(plan.get("overwrite", True)),
                "dry_run": bool(plan.get("dry_run", False)),
            },
            source_map=None
        )
    except Exception as e:
        msg = str(e)
        msg = (msg or "Chunking failed.")[:200]
        raise HTTPException(status_code=502, detail=f"Chunking failed: {msg}")

@router.post("/course/{course_id}/chunk")
async def chunk_course(course_id: str, plan: dict):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    pdf_ids = course.get("pdf_ids", []) or []
    markdown, source_map = await get_combined_markdown_and_offsets(pdf_ids)
    if not markdown.strip():
        raise HTTPException(status_code=422, detail="No extractable text/markdown in course PDFs")

    try:
        return await chunk_text_and_persist(
            markdown,
            {"pdf_id": None, "course_id": course_id},
            {
                "method": plan.get("method", "LLM [JSON]"),
                "target_tokens": int(plan.get("target_tokens", 900)) if plan.get("method") == "Other" else None,
                "min_tokens": int(plan.get("min_tokens", 400)) if plan.get("method") == "Other" else None,
                "max_tokens": int(plan.get("max_tokens", 1200)) if plan.get("method") == "Other" else None,
                "desired_chunks": plan.get("desired_chunks"),
                "title_strategy": plan.get("title_strategy", "firstline"),
                "overwrite": bool(plan.get("overwrite", True)),
                "dry_run": bool(plan.get("dry_run", False)),
            },
            source_map=source_map
        )
    except Exception as e:
        msg = str(e)
        msg = (msg or "Chunking failed.")[:200]
        raise HTTPException(status_code=502, detail=f"Chunking failed: {msg}")

@router.get("/chunks")
async def list_chunks(pdf_id: str | None = None, course_id: str | None = None, limit: int = 1000):
    if not pdf_id and not course_id:
        raise HTTPException(status_code=400, detail="Provide pdf_id or course_id")
    filt = {"pdf_id": pdf_id, "course_id": None} if pdf_id else {"course_id": course_id, "pdf_id": None}
    rows = await chunks_collection.find(filt).sort("index", 1).limit(max(1, min(limit, 5000))).to_list(length=5000)
    for r in rows:
        r["_id"] = str(r["_id"])
    return rows

@router.delete("/chunks")
async def delete_chunks(pdf_id: str | None = None, course_id: str | None = None):
    if not pdf_id and not course_id:
        raise HTTPException(status_code=400, detail="Provide pdf_id or course_id")
    filt = {"pdf_id": pdf_id, "course_id": None} if pdf_id else {"course_id": course_id, "pdf_id": None}
    res = await chunks_collection.delete_many(filt)
    return {"deleted": res.deleted_count}

@router.get("/chunks/export")
async def export_chunks(pdf_id: str | None = None, course_id: str | None = None):
    if not pdf_id and not course_id:
        raise HTTPException(status_code=400, detail="Provide pdf_id or course_id")
    filt = {"pdf_id": pdf_id, "course_id": None} if pdf_id else {"course_id": course_id, "pdf_id": None}
    rows = await chunks_collection.find(filt).sort("index", 1).to_list(length=10000)
    buff = io.StringIO()
    for r in rows:
        header = (r.get("meta", {}) or {}).get("header", "")
        src_idx = (r.get("meta", {}) or {}).get("source_pdf_index", None)
        buff.write(f"\n\n===== CHUNK {r['index']} | {r.get('start',0)}–{r.get('end',0)} | ~{r.get('token_estimate',0)} tok | src:{src_idx} =====\n")
        if header:
            buff.write(f"### {header}\n\n")
        buff.write(r.get("text", ""))
    buff.seek(0)
    filename = f"chunks-{'pdf-' + pdf_id if pdf_id else 'course-' + course_id}.txt"
    return StreamingResponse(
        io.BytesIO(buff.read().encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

class _RefreshRequest(BaseModel):
    persist: bool = False


@router.post("/course/{course_id}/chunks/refresh-metadata")
async def refresh_course_chunk_metadata(course_id: str, req: _RefreshRequest):
    rows = await chunks_collection.find({"course_id": course_id}).sort("index", 1).to_list(length=5000)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks for course")
    previews = [{"index": int(r.get("index", i)), "text": (r.get("text") or "")[:1200]} for i, r in enumerate(rows)]
    try:
        from app.providers import AI
        if not hasattr(AI, "generate_chunk_metadata"):
            raise HTTPException(status_code=501, detail="Provider does not support metadata generation")
        enriched = AI.generate_chunk_metadata(previews)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="LLM metadata generation failed")

    if req.persist:
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for md in enriched:
            try:
                idx = int(md.get("chunk_index")) if isinstance(md, dict) else None
            except Exception:
                idx = None
            if idx is None:
                continue
            extra = dict(md)
            extra["enriched_at"] = now_iso
            try:
                from app.providers import AI as _AI
                extra["embedded_with"] = getattr(_AI, "embedding_model", None) or "text-embedding-3-small"
            except Exception:
                pass
            await chunks_collection.update_one(
                {"course_id": course_id, "index": idx},
                {"$set": {"meta.enriched": extra}},
            )
    return {"metadata": enriched}


@router.post("/pdf/{pdf_id}/chunks/refresh-metadata")
async def refresh_pdf_chunk_metadata(pdf_id: str, req: _RefreshRequest):
    rows = await chunks_collection.find({"pdf_id": pdf_id}).sort("index", 1).to_list(length=5000)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks for PDF")
    previews = [{"index": int(r.get("index", i)), "text": (r.get("text") or "")[:1200]} for i, r in enumerate(rows)]
    try:
        from app.providers import AI
        if not hasattr(AI, "generate_chunk_metadata"):
            raise HTTPException(status_code=501, detail="Provider does not support metadata generation")
        enriched = AI.generate_chunk_metadata(previews)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="LLM metadata generation failed")

    if req.persist:
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for md in enriched:
            try:
                idx = int(md.get("chunk_index")) if isinstance(md, dict) else None
            except Exception:
                idx = None
            if idx is None:
                continue
            extra = dict(md)
            extra["enriched_at"] = now_iso
            try:
                from app.providers import AI as _AI
                extra["embedded_with"] = getattr(_AI, "embedding_model", None) or "text-embedding-3-small"
            except Exception:
                pass
            await chunks_collection.update_one(
                {"pdf_id": pdf_id, "index": idx},
                {"$set": {"meta.enriched": extra}},
            )
    return {"metadata": enriched}
