import uuid
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re
from app.providers import AI
from app.providers.heuristic import split_into_paragraphs_with_spans, estimate_tokens_from_chars
from app.db import chunks_collection
from app.schemas import Chunk
from app.utils.markdown_to_json import markdown_to_structured  # <-- NEW


def buffer_and_metas_from_structured(structured: List[dict]) -> Tuple[str, List[dict]]:
    """
    Build a synthetic buffer by concatenating structured[i]["text"] with a '\n\n' separator,
    and return metas with exact (start, end) spans aligned to the structured indices.
    The metas list length equals len(structured), and metas[k] maps to structured[k].
    """
    parts: List[str] = []
    metas: List[dict] = []
    pos = 0
    sep = "\n\n"

    for i, p in enumerate(structured):
        t = p.get("text", "") or ""
        start = pos
        parts.append(t)
        pos += len(t)
        end = pos
        metas.append({"index": i, "start": start, "end": end})
        if i < len(structured) - 1:
            parts.append(sep)
            pos += len(sep)

    buffer_text = "".join(parts)
    return buffer_text, metas

def build_chunks_from_boundaries(full_text: str, metas: List[Dict[str, Any]], boundaries: List[int]) -> List[Dict[str, Any]]:
    spans = [(p["start"], p["end"]) for p in metas]
    out = []
    for idx in range(len(boundaries) - 1):
        a, b = boundaries[idx], boundaries[idx + 1]
        start = spans[a][0]
        end = spans[b - 1][1] if b - 1 >= a else spans[a][1]
        seg = full_text[start:end]
        out.append({
            "index": idx, "start": start, "end": end, "text": seg,
            "token_estimate": estimate_tokens_from_chars(len(seg)),
            "meta": {"paras_range": [a, b], "count_paras": b - a}
        })
    return out

def _title_for_chunk(text: str, strategy: str) -> str:
    strategy = (strategy or "firstline").lower()
    if strategy == "llm":
        try:
            return AI.title_from_text(text) or "Section"
        except Exception:
            pass
    if strategy == "keywords":
        return AI.__class__.title_from_text(AI, text)
    return AI.__class__.title_from_text(AI, text)

def _annotate_source(source_map: Optional[List[Tuple[int, int, str, int]]], start: int, end: int) -> Dict[str, Any]:
    source_pdf_id, source_pdf_index = None, None
    if source_map:
        best = 0
        for (s, e, pid, idx) in source_map:
            ov = max(0, min(end, e) - max(start, s))
            if ov > best:
                best = ov
                source_pdf_id, source_pdf_index = pid, idx
    return {"source_pdf_id": source_pdf_id, "source_pdf_index": source_pdf_index}

async def chunk_text_and_persist(
    markdown_text: str,
    target_key: dict,
    plan: dict,
    source_map: Optional[List[Tuple[int, int, str, int]]] = None
):
    metas = split_into_paragraphs_with_spans(markdown_text)

    method = (plan.get("method") or "LLM [JSON]").strip()
    desired_chunks = plan.get("desired_chunks")
    target_tokens = int(plan.get("target_tokens", 900)) if plan.get("target_tokens") is not None else 900
    min_tokens = int(plan.get("min_tokens", 400)) if plan.get("min_tokens") is not None else 400
    max_tokens = int(plan.get("max_tokens", 1200)) if plan.get("max_tokens") is not None else 1200
    title_strategy = (plan.get("title_strategy") or "firstline").lower()

    enriched_meta_for_chunks = None
    if method in ("LLM [JSON]", "LLM [PDF]"):
        structured = markdown_to_structured(markdown_text)

        use_combined = True
        if use_combined and hasattr(AI, "propose_boundaries_with_metadata"):
            try:
                both = AI.propose_boundaries_with_metadata(structured, desired_chunks=None, strict_count=False)
                boundaries = both.get("boundaries") or []
                enriched_meta_for_chunks = both.get("metadata") or None
                try:
                    from pathlib import Path
                    import json as _json
                    from datetime import datetime as _dt
                    dbg_dir = Path(os.getenv("CHUNK_DEBUG_DIR", "debug")) / "chunking"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    stamp = _dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    (dbg_dir / f"combined_response_{stamp}.json").write_text(_json.dumps(both, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception as e:
                raise RuntimeError(f"Combined segmentation error: {str(e)}")
        else:
            raise RuntimeError("Chunking method not available.")

        buffer_text, metas_for_boundaries = buffer_and_metas_from_structured(structured)

    elif method == "LLM [Raw]":
        metas_for_boundaries = split_into_paragraphs_with_spans(markdown_text)
        boundaries = AI.propose_boundaries_llm_raw_exact(metas_for_boundaries, markdown_text, int(desired_chunks or 0))
        buffer_text = markdown_text

    elif method == "Context":
        metas_for_boundaries = split_into_paragraphs_with_spans(markdown_text)
        boundaries = AI.propose_boundaries_context(metas_for_boundaries, markdown_text, target_tokens or 900, min_tokens or 400, max_tokens or 1200)
        buffer_text = markdown_text

    elif method == "Other":
        metas_for_boundaries = split_into_paragraphs_with_spans(markdown_text)
        boundaries = AI.propose_boundaries(metas_for_boundaries, target_tokens or 900, min_tokens or 400, max_tokens or 1200)
        buffer_text = markdown_text

    else:
        structured = markdown_to_structured(markdown_text)
        boundaries = AI.propose_boundaries_llm_json_exact(structured, int(desired_chunks or 0))
        buffer_text, metas_for_boundaries = buffer_and_metas_from_structured(structured)


    chunks = build_chunks_from_boundaries(buffer_text, metas_for_boundaries, boundaries)


    if plan.get("dry_run"):
        preview = []
        for c in chunks[:50]:
            title = _title_for_chunk(c["text"], title_strategy)
            src = _annotate_source(source_map, c["start"], c["end"])
            preview.append({
                "index": c["index"], "start": c["start"], "end": c["end"],
                "token_estimate": c["token_estimate"],
                "title": title, **src,
                "preview": c["text"][:200].replace("\n", " ")
            })
        return {"paragraphs": len(metas), "chunks": len(chunks), "preview": preview, "method": method}

    if plan.get("overwrite"):
        await chunks_collection.delete_many(target_key)

    now = datetime.utcnow()
    docs = []
    if enriched_meta_for_chunks is None:
        enriched_meta_for_chunks = None

    idx_meta_map = None
    if enriched_meta_for_chunks and isinstance(enriched_meta_for_chunks, list):
        try:
            idx_meta_map = {int(m.get("chunk_index")): m for m in enriched_meta_for_chunks if isinstance(m, dict) and m.get("chunk_index") is not None}
        except Exception:
            idx_meta_map = None

    for c in chunks:
        title = _title_for_chunk(c["text"], title_strategy)
        src = _annotate_source(source_map, c["start"], c["end"])
        meta = {**c.get("meta", {}), "header": title, **src, "method": method}
        if enriched_meta_for_chunks and isinstance(enriched_meta_for_chunks, list):
            i = c["index"]
            enriched = None
            if idx_meta_map is not None and i in idx_meta_map:
                enriched = dict(idx_meta_map.get(i) or {})
            elif 0 <= i < len(enriched_meta_for_chunks):
                enriched = dict(enriched_meta_for_chunks[i] or {})
            if enriched is not None:
                enriched["embedded_with"] = getattr(AI, "embedding_model", None) or "text-embedding-3-small"
                enriched["enriched_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                meta["enriched"] = enriched
        docs.append(Chunk(
            id=str(uuid.uuid4()),
            pdf_id=target_key.get("pdf_id"),
            course_id=target_key.get("course_id"),
            index=c["index"], start=c["start"], end=c["end"], text=c["text"],
            token_estimate=c["token_estimate"], meta=meta, created_at=now
        ).dict())
    if docs:
        await chunks_collection.insert_many(docs)
    return {"chunks": len(docs), "method": method}
