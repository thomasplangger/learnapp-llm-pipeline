from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from fastapi import APIRouter, HTTPException
from datetime import datetime
import os
import re

from app.db import chunks_collection, courses_collection
from app.providers import AI

router = APIRouter(tags=["learning-objectives"])


@dataclass
class _DPCell:
    cost: float
    prev: int


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "with", "without", "into", "over",
    "under", "between", "about", "after", "before", "than", "then", "these", "those",
    "their", "they", "we", "you", "your", "our", "its", "can", "may", "might", "will",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if t and t not in _STOPWORDS]


def _keyword_list(raw: str) -> List[str]:
    parts = re.split(r"[;,]", raw or "")
    out = []
    for p in parts:
        p = " ".join(p.strip().split())
        if p:
            out.append(p)
    return out


def _heuristic_lo_metadata(items: List[Dict[str, str]], fallback_title: str | None = None) -> Dict[str, Any]:
    titles = [str(it.get("title") or "").strip() for it in items if str(it.get("title") or "").strip()]
    summaries = [str(it.get("summary") or "").strip() for it in items if str(it.get("summary") or "").strip()]
    keywords_raw = []
    for it in items:
        kw = str(it.get("keywords") or "").strip()
        if kw:
            keywords_raw.extend(_keyword_list(kw))

    corpus = " ".join(titles + summaries)
    freq = Counter(_tokenize(corpus))

    best_title = None
    best_score = -1
    for t in titles:
        score = sum(freq.get(tok, 0) for tok in _tokenize(t))
        if score > best_score:
            best_score = score
            best_title = t
    title = (best_title or fallback_title or "Learning Objective").strip()[:90]

    summary = ""
    if summaries:
        sentences = []
        for s in summaries:
            parts = re.split(r"(?<=[.!?])\s+", s)
            sentences.extend([p.strip() for p in parts if p.strip()])
        scored = []
        for s in sentences:
            scored.append((sum(freq.get(tok, 0) for tok in _tokenize(s)), s))
        scored.sort(key=lambda x: (-x[0], -len(x[1])))
        top = [s for _, s in scored[:2]]
        summary = " ".join(top).strip()
        if len(summary) > 650:
            summary = summary[:649].rstrip() + "..."

    kw_freq = Counter([k.lower() for k in keywords_raw if k])
    top_keywords = []
    for kw, _ in kw_freq.most_common(8):
        top_keywords.append(kw)
    top_keywords = [k.strip() for k in top_keywords if k.strip()][:12]
    verbs = ["Explain", "Identify", "Apply", "Compare", "Summarize"]
    verb_offset = abs(hash(title)) % len(verbs)
    objectives = [f"{verbs[(i + verb_offset) % len(verbs)]} {kw}." for i, kw in enumerate(top_keywords[:5])]
    objectives = [o.strip() for o in objectives if o.strip()][:5]

    return {
        "title": title,
        "summary": summary,
        "objectives": objectives,
        "keywords": top_keywords,
    }


# def _segment_cost(prefix_sum: List[List[float]], prefix_sq: List[float], i: int, j: int) -> float:
#     """Squared error of segment [i, j) using vector mean: SSE = sum||x||^2 - n*||mean||^2."""
#     n = j - i
#     if n <= 0:
#         return 0.0
#     dim = len(prefix_sum[0])
#     S = [prefix_sum[j][d] - prefix_sum[i][d] for d in range(dim)]
#     sumsq = prefix_sq[j] - prefix_sq[i]
#     mean_norm2 = 0.0
#     for d in range(dim):
#         mean_norm2 += (S[d] / n) * (S[d] / n)
#     return max(0.0, float(sumsq - n * mean_norm2))


# def _prefix_stats(vecs: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
#     if not vecs:
#         return [[0.0]], [0.0]
#     dim = len(vecs[0])
#     P = [[0.0] * dim for _ in range(len(vecs) + 1)]
#     Q = [0.0] * (len(vecs) + 1)
#     for i, v in enumerate(vecs, start=1):
#         for d in range(dim):
#             P[i][d] = P[i - 1][d] + float(v[d])
#         Q[i] = Q[i - 1] + sum(float(x) * float(x) for x in v)
#     return P, Q


# def _split_contiguous(vecs: List[List[float]], N: int) -> List[int]:
#     """Return boundaries as indices [0=...=M], exactly N segments, minimizing SSE."""
#     M = len(vecs)
#     if M == 0:
#         return [0, 0]
#     N = max(1, min(N, M))
#     P, Q = _prefix_stats(vecs)
#     # dp[k][j] = min cost to split first j items into k segments
#     dp: List[List[_DPCell]] = [[_DPCell(float("inf"), -1) for _ in range(M + 1)] for _ in range(N + 1)]
#     dp[0][0] = _DPCell(0.0, -1)
#     for j in range(1, M + 1):
#         dp[1][j] = _DPCell(_segment_cost(P, Q, 0, j), 0)
#     for k in range(2, N + 1):
#         for j in range(k, M + 1):  # at least one per segment
#             best = _DPCell(float("inf"), -1)
#             for i in range(k - 1, j):
#                 c = dp[k - 1][i].cost + _segment_cost(P, Q, i, j)
#                 if c < best.cost:
#                     best = _DPCell(c, i)
#             dp[k][j] = best
#     # backtrack
#     bounds = [M]
#     k, j = N, M
#     while k > 0 and j >= 0:
#         i = dp[k][j].prev
#         if i < 0:
#             break
#         bounds.append(i)
#         j = i
#         k -= 1
#     bounds.append(0)
#     bounds = sorted(set(bounds))
#     out = [bounds[0]]
#     for x in bounds[1:]:
#         if x > out[-1]:
#             out.append(x)
#     if out[0] != 0:
#         out = [0] + out
#     if out[-1] != M:
#         out[-1] = M
#     return out


@router.get("/course/{course_id}/learning-objectives")
async def get_learning_objectives(course_id: str):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    los = course.get("learning_objectives") or []
    return {"groups": los}


@router.post("/course/{course_id}/learning-objectives")
async def build_learning_objectives(course_id: str, body: Dict[str, Any]):
    rows = await chunks_collection.find({"course_id": course_id}).sort("index", 1).to_list(length=5000)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks for course")
    texts = [(r.get("text") or "")[:2000] for r in rows]
    vecs = []
    try:
        vecs = AI.batch_embed_texts(texts)
    except Exception:
        vecs = [[] for _ in texts]

    missing_idx = [i for i, v in enumerate(vecs) if not v]
    if missing_idx:
        metas_txt = []
        for i in missing_idx:
            enr = ((rows[i].get("meta", {}) or {}).get("enriched", {}) or {})
            emb = enr.get("embedding") or []
            if emb:
                vecs[i] = emb
                metas_txt.append("")
            else:
                t = "\n".join(filter(None, [enr.get("title"), enr.get("summary"), ", ".join(enr.get("keywords", []))]))
                metas_txt.append(t)
        to_embed = [metas_txt[idx] for idx in range(len(metas_txt))]
        if any(to_embed):
            try:
                meta_vecs = AI.batch_embed_texts([t for t in to_embed])
            except Exception:
                meta_vecs = [[] for _ in to_embed]
            k = 0
            for i in missing_idx:
                if not vecs[i]:
                    vecs[i] = meta_vecs[k] if k < len(meta_vecs) else []
                k += 1

    dim = next((len(v) for v in vecs if v), 0)
    if dim == 0:
        raise HTTPException(status_code=422, detail="Embeddings unavailable for grouping")
    for i, v in enumerate(vecs):
        if not v:
            vecs[i] = [0.0] * dim
        elif len(v) != dim:
            vv = list(v)[:dim]
            if len(vv) < dim:
                vv += [0.0] * (dim - len(vv))
            vecs[i] = vv

    method = (body.get("method") or "heuristic").lower().strip() if body else "heuristic"
    persist = bool((body or {}).get("persist", False))
    N: Optional[int] = None
    try:
        N = int(body.get("N")) if body and body.get("N") is not None else None
    except Exception:
        N = None
    if N is None:
        topics = []
        for r in rows:
            enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
            if enr.get("topic"):
                topics.append(str(enr.get("topic")))
            elif enr.get("keywords"):
                topics.extend([str(k) for k in (enr.get("keywords") or [])])
        if hasattr(AI, "choose_lo_count"):
            try:
                N = AI.choose_lo_count(topics)
            except Exception:
                N = None
        if not N:
            import math
            N = max(3, min(10, int(math.sqrt(max(1, len(rows))))))

    N = max(1, int(N))

    groups_payload: List[Dict[str, Any]] = []
    palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7"]

    if method == "llm" and hasattr(AI, "group_chunks_semantic"):
        try:
            chunks_payload = [{"index": r.get("index", i), "text": (r.get("text") or "")} for i, r in enumerate(rows)]
            refreshed = AI.generate_chunk_metadata(chunks_payload)
            if refreshed:
                for m in refreshed:
                    idx = int(m.get("chunk_index", -1))
                    if 0 <= idx < len(rows):
                        meta = rows[idx].get("meta") or {}
                        meta["enriched"] = m
                        rows[idx]["meta"] = meta
        except Exception:
            pass
        items_for_llm: List[Dict[str, Any]] = []
        for i, r in enumerate(rows):
            enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
            items_for_llm.append({
                "index": i,
                "title": enr.get("title") or (r.get("meta", {}) or {}).get("header") or "",
                "topic": enr.get("topic") or "",
                "keywords": (enr.get("keywords") or [])[:10],
                "summary": enr.get("summary") or "",
            })
        try:
            grouped = AI.group_chunks_semantic(items_for_llm, desired_N=N)
        except Exception:
            grouped = {"groups": []}
        llm_groups = grouped.get("groups") or []
        for gi, g in enumerate(llm_groups):
            members_idx = [int(x) for x in (g.get("members") or []) if 0 <= int(x) < len(rows)]
            members = [rows[j] for j in members_idx]
            items = []
            for r in members[:10]:
                enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
                items.append({
                    "title": enr.get("title") or (r.get("meta", {}) or {}).get("header") or "",
                    "summary": enr.get("summary") or "",
                    "keywords": ", ".join((enr.get("keywords") or [])),
                })
            lo_data = None
            if os.getenv("LO_REFINE_SUMMARY", "1") == "1" and hasattr(AI, "summarize_learning_objective") and items:
                try:
                    lo_data = AI.summarize_learning_objective(items)
                except Exception:
                    lo_data = None
            if not lo_data and items:
                lo_data = _heuristic_lo_metadata(items, fallback_title=f"Learning Objective {gi+1}")
            title = (g.get("title") or (lo_data or {}).get("title") or f"Learning Objective {gi+1}")
            summary = (g.get("summary") or (lo_data or {}).get("summary") or "")
            keyword_tags = (g.get("keywords") or (lo_data or {}).get("keywords") or [])
            groups_payload.append({
                "index": gi,
                "title": title,
                "summary": summary,
                "objectives": (lo_data or {}).get("objectives") or [],
                "keyword_tags": keyword_tags,
                "chunk_indices": members_idx,
                "color": palette[gi % len(palette)],
            })
        if groups_payload:
            if persist:
                try:
                    await courses_collection.update_one(
                        {"id": course_id},
                        {"$set": {
                            "learning_objectives": groups_payload,
                            "learning_objectives_updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }}
                    )
                except Exception:
                    pass
            return {"N": len(groups_payload), "groups": groups_payload}

    if not N:
        K = max(2, round(len(rows) / 3))
    else:
        K = max(1, int(N))

    import random, math

    def _normalize(v: List[float]) -> List[float]:
        s = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/s for x in v]

    X = [_normalize(v) for v in vecs]
    M = len(X)
    K = min(max(1, K), M)
    random.seed(42)
    cents = [X[i] for i in sorted(random.sample(range(M), K))]

    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    labels = [0]*M
    for _ in range(10):
        changed = False
        for i, v in enumerate(X):
            j = max(range(K), key=lambda k: _dot(v, cents[k]))
            if labels[i] != j:
                labels[i] = j; changed = True
        sums = [[0.0]*len(X[0]) for _ in range(K)]
        counts = [0]*K
        for i, v in enumerate(X):
            j = labels[i]
            counts[j] += 1
            for d in range(len(v)):
                sums[j][d] += v[d]
        for k in range(K):
            if counts[k] > 0:
                cents[k] = _normalize([s/counts[k] for s in sums[k]])
        if not changed:
            break

    label_to_indices: Dict[int, List[int]] = {k: [] for k in range(K)}
    for i, lab in enumerate(labels):
        label_to_indices[lab].append(i)

    gi = 0
    for k in range(K):
        idxs = label_to_indices[k]
        if not idxs:
            continue
        members = [rows[j] for j in idxs]
        items = []
        for r in members[:10]:
            enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
            items.append({
                "title": enr.get("title") or (r.get("meta", {}) or {}).get("header") or "",
                "summary": enr.get("summary") or "",
                "keywords": ", ".join((enr.get("keywords") or [])),
            })
        lo_data = None
        if os.getenv("LO_REFINE_SUMMARY", "1") == "1" and hasattr(AI, "summarize_learning_objective") and items:
            try:
                lo_data = AI.summarize_learning_objective(items)
            except Exception:
                lo_data = None
        if not lo_data and items:
            lo_data = _heuristic_lo_metadata(items, fallback_title=f"Learning Objective {gi+1}")
        if not lo_data:
            lo_data = {"title": f"Learning Objective {gi+1}", "summary": "", "objectives": [], "keywords": []}
        groups_payload.append({
            "index": gi,
            "title": lo_data.get("title") or f"Learning Objective {gi+1}",
            "summary": lo_data.get("summary") or "",
            "objectives": lo_data.get("objectives") or [],
            "keyword_tags": lo_data.get("keywords") or [],
            "chunk_indices": idxs,
            "color": palette[gi % len(palette)],
        })
        gi += 1

    if persist:
        try:
            await courses_collection.update_one(
                {"id": course_id},
                {"$set": {
                    "learning_objectives": groups_payload,
                    "learning_objectives_updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }}
            )
        except Exception:
            pass

    return {"N": len(groups_payload), "groups": groups_payload}

def _ensure_unique_members(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    for g in groups:
        idxs = []
        for x in (g.get("chunk_indices") or []):
            i = int(x)
            if i not in seen:
                seen.add(i)
                idxs.append(i)
        g["chunk_indices"] = idxs
    return groups


@router.patch("/course/{course_id}/learning-objectives")
async def patch_learning_objective(course_id: str, body: Dict[str, Any]):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    groups = list(course.get("learning_objectives") or [])
    if not isinstance(groups, list) or not groups:
        raise HTTPException(status_code=404, detail="No learning objectives to patch")

    lo_index = body.get("lo_index")
    if lo_index is None or not (0 <= int(lo_index) < len(groups)):
        raise HTTPException(status_code=422, detail="Invalid lo_index")
    lo = dict(groups[int(lo_index)])

    if "title" in body:
        lo["title"] = str(body.get("title") or "").strip()
    if "summary" in body:
        lo["summary"] = str(body.get("summary") or "")
    if "objectives" in body:
        objs = body.get("objectives") or []
        lo["objectives"] = [str(x) for x in objs if str(x).strip()]
    if "keyword_tags" in body:
        tags = body.get("keyword_tags") or []
        lo["keyword_tags"] = [str(x) for x in tags if str(x).strip()]
    if "chunk_indices" in body:
        new_idxs = [int(x) for x in (body.get("chunk_indices") or [])]
        lo["chunk_indices"] = new_idxs
        for j, g in enumerate(groups):
            if j == int(lo_index):
                continue
            g["chunk_indices"] = [k for k in (g.get("chunk_indices") or []) if k not in new_idxs]

    groups[int(lo_index)] = lo
    groups = _ensure_unique_members(groups)

    await courses_collection.update_one(
        {"id": course_id},
        {"$set": {"learning_objectives": groups, "learning_objectives_updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}}
    )
    return {"groups": groups}


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5 or 1.0
    nb = sum(y*y for y in b) ** 0.5 or 1.0
    return float(s / (na * nb))


@router.post("/course/{course_id}/learning-objectives/suggest-members")
async def suggest_lo_members(course_id: str, body: Dict[str, Any]):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    groups = list(course.get("learning_objectives") or [])
    if not groups:
        raise HTTPException(status_code=404, detail="No learning objectives to suggest for")
    lo_index = int(body.get("lo_index", 0))
    if not (0 <= lo_index < len(groups)):
        raise HTTPException(status_code=422, detail="Invalid lo_index")

    rows = await chunks_collection.find({"course_id": course_id}).sort("index", 1).to_list(length=5000)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks for course")
    target_size = int(body.get("N") or len(groups[lo_index].get("chunk_indices") or []) or 6)

    dim = None
    embeds: List[List[float]] = []
    need_texts: List[str] = []
    for r in rows:
        enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
        vec = enr.get("embedding") or []
        if isinstance(vec, list) and vec:
            embeds.append([float(x) for x in vec])
            dim = dim or len(vec)
            need_texts.append("")
        else:
            t = "\n".join(filter(None, [
                (enr.get("title") or (r.get("meta", {}) or {}).get("header") or ""),
                enr.get("summary") or "",
                ", ".join((enr.get("keywords") or []))
            ]))
            need_texts.append(t)
            embeds.append([])

    if any(not v for v in embeds):
        try:
            new_vecs = AI.batch_embed_texts([t for t in need_texts])
        except Exception:
            new_vecs = [[0.0] for _ in need_texts]
        for i in range(len(embeds)):
            if not embeds[i] and i < len(new_vecs):
                embeds[i] = [float(x) for x in (new_vecs[i] or [])]
                dim = dim or (len(embeds[i]) if embeds[i] else None)

    query_title = str(body.get("title") or groups[lo_index].get("title") or "").strip()
    if not query_title:
        raise HTTPException(status_code=422, detail="Provide a title to suggest members")
    try:
        qvecs = AI.batch_embed_texts([query_title])
        qv = [float(x) for x in (qvecs[0] if qvecs else [])]
    except Exception:
        qv = []

    scores: List[Tuple[int, float]] = []
    for idx, v in enumerate(embeds):
        s = _cosine(qv, v) if qv and v else 0.0
        scores.append((idx, s))
    assigned = set()
    for gi, g in enumerate(groups):
        if gi == lo_index:
            continue
        for ii in (g.get("chunk_indices") or []):
            assigned.add(int(ii))
    scores = [(i, s) for (i, s) in scores if i not in assigned]
    scores.sort(key=lambda t: t[1], reverse=True)

    suggested = scores[:max(1, target_size)]
    return {"lo_index": lo_index, "suggested_indices": [i for (i, _) in suggested], "scores": suggested}


@router.post("/course/{course_id}/learning-objectives/regenerate")
async def regenerate_lo_text(course_id: str, body: Dict[str, Any]):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    groups = list(course.get("learning_objectives") or [])
    if not groups:
        raise HTTPException(status_code=404, detail="No learning objectives to regenerate")
    lo_index = int(body.get("lo_index", 0))
    if not (0 <= lo_index < len(groups)):
        raise HTTPException(status_code=422, detail="Invalid lo_index")
    mode = (body.get("mode") or "all").lower()

    members = groups[lo_index].get("chunk_indices") or []
    rows = await chunks_collection.find({"course_id": course_id}).sort("index", 1).to_list(length=5000)
    items = []
    for i in members[:10]:
        if 0 <= i < len(rows):
            r = rows[i]
            enr = ((r.get("meta", {}) or {}).get("enriched", {}) or {})
            items.append({
                "title": enr.get("title") or (r.get("meta", {}) or {}).get("header") or "",
                "summary": enr.get("summary") or "",
                "keywords": ", ".join((enr.get("keywords") or [])),
            })

    updated = dict(groups[lo_index])
    lo_data = None
    if hasattr(AI, "summarize_learning_objective") and items:
        try:
            lo_data = AI.summarize_learning_objective(items)
        except Exception:
            lo_data = None
    if not lo_data and items:
        lo_data = _heuristic_lo_metadata(items, fallback_title=updated.get("title") or f"Learning Objective {lo_index+1}")
    if lo_data:
        if mode in ("title", "all"):
            updated["title"] = lo_data.get("title") or updated.get("title")
        if mode in ("summary", "all"):
            updated["summary"] = lo_data.get("summary") or updated.get("summary")
        if mode in ("objectives", "all"):
            updated["objectives"] = lo_data.get("objectives") or updated.get("objectives")

    persist = bool(body.get("persist", True))
    if persist:
        groups[lo_index] = updated
        await courses_collection.update_one(
            {"id": course_id},
            {"$set": {"learning_objectives": groups, "learning_objectives_updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}}
        )
    return {"group": updated}


@router.post("/course/{course_id}/learning-objectives/move-chunk")
async def move_chunk_between_los(course_id: str, body: Dict[str, Any]):
    course = await courses_collection.find_one({"id": course_id})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    groups = list(course.get("learning_objectives") or [])
    if not groups:
        raise HTTPException(status_code=404, detail="No learning objectives to edit")
    ci = int(body.get("chunk_index"))
    to_lo = int(body.get("to_lo_index"))
    if not (0 <= to_lo < len(groups)):
        raise HTTPException(status_code=422, detail="Invalid to_lo_index")
    pos = body.get("position")
    for g in groups:
        g["chunk_indices"] = [k for k in (g.get("chunk_indices") or []) if int(k) != ci]
    dest = groups[to_lo].get("chunk_indices") or []
    if isinstance(pos, int) and 0 <= pos <= len(dest):
        dest = dest[:pos] + [ci] + dest[pos:]
    else:
        dest.append(ci)
    groups[to_lo]["chunk_indices"] = dest
    groups = _ensure_unique_members(groups)
    groups = [g for g in groups if (g.get("chunk_indices") or [])]
    await courses_collection.update_one(
        {"id": course_id},
        {"$set": {"learning_objectives": groups, "learning_objectives_updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}}
    )
    return {"groups": groups}
