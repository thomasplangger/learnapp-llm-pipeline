from __future__ import annotations

import asyncio
import csv
import json
import math
import random
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata, re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.utils.create_test_data import create_test_data
from app.utils.pdf_to_markdown import pdf_bytes_to_markdown
from app.utils.markdown_to_json import markdown_to_structured
from app.providers import AI
import app.utils.evaluate_chunking as evalmod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

router = APIRouter(tags=["auto-test"])

                                    
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    parts = pp.parts
    if parts and parts[0].lower() == "backend":
        return (_project_root() / pp).resolve()
    return (_project_root() / "backend" / pp).resolve()

BASE_OUTDIR = _resolve_path("backend/testdata")
SOURCE_DIR  = _resolve_path("backend/generated")
PDF_CACHE_DIR = BASE_OUTDIR / "pdf_cache"

                                 
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()
MAX_CONCURRENCY = 3                                                   

_norm_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]", re.UNICODE)

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = _punct.sub(" ", s)
    s = _norm_ws.sub(" ", s).strip()
    return s

def _ts_dir() -> str:
    return datetime.now().strftime("%d_%m_%y_%H-%M")

def _canonical_cache_name(raw: str) -> str:
    raw = (raw or "").strip()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    slug = slug.strip("_")
    if not slug:
        slug = f"cache_{_ts_dir()}"
    return slug[:80]

def _canonical_batch_label(raw: Optional[str]) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return _canonical_cache_name(raw)

def _batch_suffix(cfg: StartPayload) -> Optional[str]:
    parts = []
    if cfg.batch_label:
        parts.append(cfg.batch_label)
    if cfg.batch_variant:
        parts.append(cfg.batch_variant)
    if not parts:
        return None
    return "_".join(parts)

def _write_testrun_meta(testrun_dir: Path, cfg: StartPayload, runs_total: int, plan_docs: List[int], mods_cfg: Dict[str, Any]) -> None:
    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "testrun_dir": str(testrun_dir.resolve()),
        "testrun_name": testrun_dir.name,
        "runs_planned": runs_total,
        "docs_per_run": [int(x) for x in plan_docs],
        "min_pages": int(cfg.min_pages),
        "max_pages": int(cfg.max_pages),
        "min_pdfs": int(cfg.min_pdfs),
        "max_pdfs": int(cfg.max_pdfs),
        "seed": cfg.seed,
        "selection_mode": cfg.selection_mode,
        "length_mode": cfg.length_mode,
        "topics": cfg.topics or [],
        "cache_name": cfg.cache_name,
        "use_cached": cfg.use_cached,
        "batch_label": cfg.batch_label,
        "batch_variant": cfg.batch_variant,
        "mods": mods_cfg or {},
    }
    try:
        suffix = _batch_suffix(cfg)
        name = f"testrun_meta_{suffix}.json" if suffix else "testrun_meta.json"
        (testrun_dir / name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

def _cache_root(cache_name: str, *, ensure: bool = False) -> Path:
    root = PDF_CACHE_DIR / cache_name
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
    return root

def _cache_destination(cache_name: str) -> Path:
    return _cache_root(cache_name, ensure=True)

def _cache_run_artifacts(run_dir: Path, cache_name: str, run_index: int) -> Path:
    target_root = _cache_destination(cache_name)
    dest = target_root / f"run_{run_index}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_dir, dest)
    return dest

def _write_cache_manifest(cache_name: str, cfg: Dict[str, Any], plan_docs: List[int], testrun_dir: Path) -> None:
    root = _cache_destination(cache_name)
    manifest = {
        "cache_name": cache_name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "runs": int(cfg.get("runs", 0)),
        "docs_per_run": [int(x) for x in plan_docs],
        "source_testrun_dir": str(testrun_dir.resolve()),
        "config": cfg,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

class StartPayload(BaseModel):
    course_id: str
    runs: int = Field(..., ge=1)
    min_pages: int = Field(..., ge=1)
    max_pages: int = Field(..., ge=1)
    min_pdfs: int = Field(..., ge=1)
    max_pdfs: int = Field(..., ge=1)
    seed: Optional[int] = None
    selection_mode: Optional[str] = "mixed"
    length_mode: Optional[str] = "mixed"                                         
    topics: Optional[List[str]] = None                                        
    cache_only: bool = False
    cache_name: Optional[str] = None                                                           
    use_cached: Optional[str] = None                                                 
    batch_label: Optional[str] = None                                  
    batch_variant: Optional[str] = None                                     
                                                              
    mod_window_size: Optional[int] = None
    mod_heading_strength: Optional[float] = None
    mod_similarity_threshold: Optional[float] = None
    mod_hierarchical: bool = False
    mod_prechunking: bool = False
    mod_prechunking_min_pages: Optional[int] = None
    mod_merge_small: Optional[int] = None
    mod_split_large: Optional[int] = None

def _extract_mods(cfg: StartPayload) -> Dict[str, Any]:
    mods: Dict[str, Any] = {}
    if cfg.mod_window_size:
        mods["window_size"] = int(cfg.mod_window_size)
    if cfg.mod_heading_strength:
        mods["heading_strength"] = float(cfg.mod_heading_strength)
    if cfg.mod_similarity_threshold:
        mods["similarity_threshold"] = float(cfg.mod_similarity_threshold)
    if cfg.mod_hierarchical:
        mods["hierarchical"] = True
    if cfg.mod_prechunking:
        mods["prechunking"] = True
        if cfg.mod_prechunking_min_pages:
            mods["prechunking_min_pages"] = int(cfg.mod_prechunking_min_pages)
    if cfg.mod_merge_small:
        mods["merge_small"] = int(cfg.mod_merge_small)
    if cfg.mod_split_large:
        mods["split_large"] = int(cfg.mod_split_large)
    return mods

@router.get("/topics")
async def list_topics(base: str = Query("backend/generated")):
    base_path = _resolve_path(base)
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Base path not found: {base_path}")
    topics: List[str] = [p.name for p in sorted(base_path.iterdir()) if p.is_dir()]
    return {"base": str(base_path), "topics": topics}

@router.post("/auto-test/run")
async def start_auto_test(payload: StartPayload):
    if payload.min_pages > payload.max_pages:
        raise HTTPException(status_code=422, detail="min_pages must be ≤ max_pages")
    if payload.min_pdfs > payload.max_pdfs:
        raise HTTPException(status_code=422, detail="min_pdfs must be ≤ max_pdfs")
    if payload.length_mode not in (None, "mixed", "short", "medium", "long"):
        raise HTTPException(status_code=422, detail="length_mode must be one of mixed/short/medium/long")
    updates: Dict[str, Any] = {}
    if payload.cache_name:
        updates["cache_name"] = _canonical_cache_name(payload.cache_name)
    if payload.use_cached:
        updates["use_cached"] = _canonical_cache_name(payload.use_cached)
    label = _canonical_batch_label(payload.batch_label)
    variant = _canonical_batch_label(payload.batch_variant)
    if label:
        updates["batch_label"] = label
    if variant:
        updates["batch_variant"] = variant
    if updates:
        payload = payload.model_copy(update=updates)
    if payload.cache_only and not payload.cache_name:
        raise HTTPException(status_code=422, detail="cache_name must be provided when cache_only is true")
    if payload.cache_only and payload.use_cached:
        raise HTTPException(status_code=422, detail="Cannot request cache_only while also using a cached dataset")
    mods_cfg = _extract_mods(payload)
    cache_dir_for_job = None
    if payload.cache_name:
        cache_dir_for_job = str(_cache_destination(payload.cache_name).resolve())
    if payload.use_cached:
        cache_root = _cache_root(payload.use_cached)
        if not cache_root.exists():
            raise HTTPException(status_code=404, detail=f"Cached dataset '{payload.use_cached}' not found")
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DIR.exists():
        raise HTTPException(status_code=404, detail=f"SOURCE_DIR not found: {SOURCE_DIR}")

    job_id = str(uuid.uuid4())
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "starting",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "completed": 0,
            "total": int(payload.runs),
            "docs_done": 0,
            "docs_total": 0,
            "current": "",
            "phase": "",
            "testrun_dir": None,
            "error": None,
            "config": payload.model_dump(),
            "eta_seconds": None,
            "started_at": time.time(),
            "run_started": 0,
            "run_finished": 0,
            "cache_dir": cache_dir_for_job,
            "use_cached": payload.use_cached,
            "mods": mods_cfg,
        }
    asyncio.create_task(_auto_test_worker(job_id, payload))
    return {"job_id": job_id}

@router.get("/auto-test/status")
async def get_status(job_id: str = Query(...)):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        eta = job.get("eta_seconds")
        eta_h = str(timedelta(seconds=int(eta))) if isinstance(eta, (int, float)) else None
        return {
            "status": job["status"],
            "completed": job["completed"],
            "total": job["total"],
            "docs_done": job.get("docs_done", 0),
            "docs_total": job.get("docs_total", 0),
            "eta_seconds": eta,
            "eta_human": eta_h,
            "current": job["current"],
            "phase": job.get("phase"),
            "testrun_dir": job["testrun_dir"],
            "error": job["error"],
            "config": job.get("config"),
            "run_started": job.get("run_started", 0),
            "run_finished": job.get("run_finished", 0),
            "cache_dir": job.get("cache_dir"),
            "use_cached": job.get("use_cached"),
            "mods": job.get("mods"),
        }

@router.get("/auto-test/caches")
async def list_cached_datasets():
    caches: List[Dict[str, Any]] = []
    if PDF_CACHE_DIR.exists():
        for cdir in sorted(PDF_CACHE_DIR.iterdir()):
            if not cdir.is_dir():
                continue
            manifest_path = cdir / "manifest.json"
            manifest: Dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}
            run_dirs = sorted([p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("run_")])
            caches.append({
                "name": cdir.name,
                "runs": manifest.get("runs") or len(run_dirs),
                "docs_per_run": manifest.get("docs_per_run"),
                "created_at": manifest.get("created_at"),
                "path": str(cdir.resolve()),
                "last_modified": datetime.fromtimestamp(cdir.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return {"base": str(PDF_CACHE_DIR.resolve()), "caches": caches}

                                          
def _find_latest(folder: Path, pattern: str) -> Optional[Path]:
    c = list(folder.glob(pattern))
    if not c: return None
    c.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return c[0]

def _collect_run_artifacts(run_dir: Path) -> Tuple[List[Path], List[Path]]:
    pdfs: List[Path] = []
    bundles: List[Path] = []
    for sub in sorted(run_dir.glob("*")):
        if not sub.is_dir():
            continue
        pdf = _find_latest(sub, "*.pdf")
        js  = _find_latest(sub, "*.json")
        if pdf: pdfs.append(pdf)
        if js:  bundles.append(js)
    return pdfs, bundles

def _merge_bundles(bundle_paths: List[Path]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "schema_version": "merged-1.0",
        "created_at": datetime.now().isoformat(),
        "generator": {"tool": "autotest"},
        "items": [],
        "totals": {"pages_measured": 0, "tokens": 0, "words": 0},
    }
    for p in bundle_paths:
        try:
            b = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        combined["items"].extend(b.get("items", []))
        t = b.get("totals", {})
        combined["totals"]["pages_measured"] += int(t.get("pages_measured", 0))
        combined["totals"]["tokens"] += int(t.get("tokens", 0))
        combined["totals"]["words"] += int(t.get("words", 0))
    combined["selection"] = {"count": len(combined["items"])}
    return combined

def _combine_markdown(pdfs: List[Path]) -> str:
    parts: List[str] = []
    for pp in pdfs:
        md = pdf_bytes_to_markdown(pp.read_bytes()).strip()
        parts.append(md)
    return "\n\n---\n\n".join(parts) + "\n"

def _size_window(M: int, desired: int) -> tuple[int, int]:
    if desired <= 0:
        desired = 1
    target = M / desired
    slack = 0.5
    min_p = max(1, int(round(target * (1.0 - slack))))
    max_p = max(min_p, int(round(target * (1.0 + slack))))
    return (min_p, max_p)

def _strict_integrity(bounds: List[int], M: int) -> Dict[str, Any]:
    strictly_increasing = all(bounds[i] < bounds[i+1] for i in range(len(bounds)-1))
    first_is_zero = (bounds[0] == 0)
    last_is_M = (bounds[-1] == M)
    dup = len(bounds) - len(set(bounds))
    issues = []
    if not strictly_increasing: issues.append("not_strictly_increasing")
    if not first_is_zero: issues.append("first_not_zero")
    if not last_is_M: issues.append("last_not_M")
    return {
        "strictly_increasing": strictly_increasing,
        "first_is_zero": first_is_zero,
        "last_is_M": last_is_M,
        "duplicate_count": dup,
        "issues": issues,
    }

def _gold_from_bundle(paras: List[Dict[str, Any]], bundle: Dict[str, Any]) -> List[int]:
                                                                                                       
    titles = evalmod.titles_from_bundle(bundle, mode="first")
    idxs: List[int] = []
    for t in titles:
        idx = evalmod.find_paragraph_index_by_title(paras, t)
        if idx is not None:
            idxs.append(idx)
    M = len(paras)
    return [0] + sorted(set(i for i in idxs if 0 < i < M)) + [M]

def _boundary_offsets(pred: List[int], gold: List[int], M: int) -> List[int]:
    P = [p for p in pred if 0 < p < M]
    G = [g for g in gold if 0 < g < M]
    out: List[int] = []
    for p in P:
        if G:
            diffs = [(p - g) for g in G]
            out.append(sorted(diffs, key=lambda d: abs(d))[0])
    return out

def _percentile(arr: List[float], q: float) -> float:
    if not arr: return 0.0
    xs = sorted(arr)
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c: return float(xs[int(k)])
    return xs[f] * (c - k) + xs[c] * (k - f)

@dataclass
class RunRow:
    run_index: int
    subpdf_count: int
    pages_total: int
    items_total: int
    desired_chunks: int
    produced_chunks: int
    paragraphs: int
    chunk_diff: int
    chunk_diff_pct: float
    chunk_size_diff_avg: float
    chunk_size_diff_pct: float
    chunk_time_sec: float
    mods_used: str
    precision: float
    recall: float
    f1: float
    mabo: float
    too_small: int
    too_big: int
    strictly_increasing: bool
    first_is_zero: bool
    last_is_M: bool
    duplicate_count: int
    topic_primary: str
    topic_diversity: int
    topic_entropy: float
    tp: int
    fp: int
    fn: int
    offset_p50: float
    offset_p90: float
    report_dir: str

                                                             
def _process_and_eval_run(run_dir: Path, mods: Optional[Dict[str, Any]] = None) -> RunRow:
    mods = mods or {}
    pdfs, bundles = _collect_run_artifacts(run_dir)
    if not pdfs or not bundles:
        raise RuntimeError(f"Missing PDFs/JSONs in {run_dir}")

    combined_bundle = _merge_bundles(bundles)
    (run_dir / "combined_bundle.json").write_text(json.dumps(combined_bundle, indent=2), encoding="utf-8")

    pages_total = int(combined_bundle.get("totals", {}).get("pages_measured", 0))
    items_total = len(combined_bundle.get("items", []))
    desired = max(1, items_total)

    topics = [(it.get("topic") or it.get("category") or "unknown") for it in combined_bundle.get("items", [])]
    topic_mix = Counter(topics)
    topic_div = len(topic_mix) or 0
    total_items = sum(topic_mix.values()) or 1
    topic_entropy = -sum((c/total_items) * math.log2(c/total_items) for c in topic_mix.values())
    primary_topic = max(topic_mix, key=topic_mix.get) if topic_mix else "unknown"

    combined_markdown = _combine_markdown(pdfs)
    paras = markdown_to_structured(combined_markdown)
    M = len(paras)

                                                                                              
    chunk_start = time.time()
    _both = AI.propose_boundaries_with_metadata(paras, desired_chunks=None, strict_count=False, mods=mods)
    bounds = _both.get("boundaries") or []
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("empty-boundaries")
    chunk_time = time.time() - chunk_start

    (run_dir / "paragraphs_used.json").write_text(json.dumps(paras, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "llm_output.context.json").write_text(json.dumps({"boundaries": bounds}, indent=2), encoding="utf-8")

    min_p, max_p = _size_window(M, desired)
    counts = [bounds[i+1] - bounds[i] for i in range(len(bounds)-1)]
    size_v = evalmod.size_violations(counts, min_p, max_p)
    integ = _strict_integrity(bounds, M)
    gold = _gold_from_bundle(paras, combined_bundle) if items_total else [0, M]
    gold_scores = evalmod.compare_to_gold(bounds, gold, tol=2) if items_total else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "mabo": 0.0, "tp": 0, "fp": 0, "fn": 0
    }
    offsets = _boundary_offsets(bounds, gold, M)
    abs_offsets = [abs(o) for o in offsets]
    offset_p50 = _percentile(abs_offsets, 0.5)
    offset_p90 = _percentile(abs_offsets, 0.9)

    out_report_dir = Path(evalmod.evaluate(
        paragraphs_path=str(run_dir / "paragraphs_used.json"),
        llm_output_path=str(run_dir / "llm_output.context.json"),
        bundle_json_path=str(run_dir / "combined_bundle.json"),
        gold_csv_path=None,
        desired_chunks=desired,
        out_dir=str(run_dir / "eval_report"),
        size_window_from_desired=True
    ))

    metrics = {
        "M_paragraphs": M,
        "N_chunks": len(bounds)-1,
        "desired_chunks": desired,
        "para_counts": counts,
        "size_window": {"min_p": min_p, "max_p": max_p},
        "size_violations": size_v,
        "integrity": integ,
        "gold": gold,
        "gold_scores": gold_scores,
        "pages_total": pages_total,
        "items_total": items_total,
        "topic_mix": dict(topic_mix),
        "topic_diversity": topic_div,
        "topic_entropy": topic_entropy,
        "primary_topic": primary_topic,
        "offsets": offsets,
        "offset_p50": offset_p50,
        "offset_p90": offset_p90,
        "chunk_time_sec": chunk_time,
        "mods_used": mods
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (run_dir / "offsets.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset", "abs_offset", "primary_topic", "subpdf_count", "pages_total", "items_total"])
        for o in offsets:
            w.writerow([o, abs(o), primary_topic, len(pdfs), pages_total, items_total])

    produced = len(bounds) - 1
    chunk_diff = abs(desired - produced)
    chunk_diff_pct = round((chunk_diff / desired), 4) if desired else 0.0
    produced_safe = produced if produced > 0 else 1
    ideal_avg = (M / desired) if desired else float(M or 1)
    actual_avg = (M / produced_safe) if produced_safe else float(M or 1)
    chunk_size_diff_avg = round(actual_avg - ideal_avg, 4)
    chunk_size_diff_pct = round((chunk_size_diff_avg / ideal_avg), 4) if ideal_avg else 0.0

    return RunRow(
        run_index=int(run_dir.name.split("_")[-1]),
        subpdf_count=len(pdfs),
        pages_total=pages_total,
        items_total=items_total,
        desired_chunks=desired,
        produced_chunks=produced,
        paragraphs=M,
        chunk_diff=int(chunk_diff),
        chunk_diff_pct=float(chunk_diff_pct),
        chunk_size_diff_avg=float(chunk_size_diff_avg),
        chunk_size_diff_pct=float(chunk_size_diff_pct),
        chunk_time_sec=round(float(chunk_time), 2),
        mods_used=json.dumps(mods, sort_keys=True),
        precision=round(float(gold_scores.get("precision", 0.0)), 4),
        recall=round(float(gold_scores.get("recall", 0.0)), 4),
        f1=round(float(gold_scores.get("f1", 0.0)), 4),
        mabo=round(float(gold_scores.get("mabo", 0.0)), 4),
        too_small=len(size_v.get("too_small", [])),
        too_big=len(size_v.get("too_big", [])),
        strictly_increasing=bool(integ["strictly_increasing"]),
        first_is_zero=bool(integ["first_is_zero"]),
        last_is_M=bool(integ["last_is_M"]),
        duplicate_count=int(integ["duplicate_count"]),
        topic_primary=primary_topic,
        topic_diversity=topic_div,
        topic_entropy=round(float(topic_entropy), 4),
        tp=int(gold_scores.get("tp", 0)),
        fp=int(gold_scores.get("fp", 0)),
        fn=int(gold_scores.get("fn", 0)),
        offset_p50=float(offset_p50),
        offset_p90=round(float(offset_p90), 4),
        report_dir=str(out_report_dir)
    )

# ---------- plotting ----------
# def _scatter(ax, x, y, xlabel, ylabel, title):
#     ax.scatter(x, y)
#     ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True)

# def _bar_with_err(ax, labels, means, stdevs, xlabel, ylabel, title):
#     x = np.arange(len(labels))
#     ax.bar(x, means, yerr=stdevs, capsize=4)
#     ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
#     ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y")

# def _box_by_group(ax, groups, values, xlabel, ylabel, title):
#     uniq = sorted(set(groups))
#     data = [[v for g, v in zip(groups, values) if g == u] for u in uniq]
#     ax.boxplot(data, labels=[str(u) for u in uniq], showfliers=False)
#     ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y")

# def _hist(ax, data, bins, xlabel, ylabel, title):
#     ax.hist(data, bins=bins); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True)

# def _cdf(ax, data, xlabel, ylabel, title):
#     xs = np.sort(np.array(data)); xs = xs if len(xs) else np.array([0])
#     ys = np.arange(1, len(xs)+1) / len(xs)
#     ax.plot(xs, ys); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True)

# def _stacked_size_bars(ax, small, inwin, big, xlabel, ylabel, title):
#     idx = np.arange(len(small))
#     ax.bar(idx, inwin, label="in-window")
#     ax.bar(idx, small, bottom=inwin, label="too small")
#     ax.bar(idx, big, bottom=np.array(inwin)+np.array(small), label="too big")
#     ax.legend(); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y")

def _load_bounds_for_run(run_dir: Path):
    paras_p = run_dir / "paragraphs_used.json"
    bundle_p = run_dir / "combined_bundle.json"
    pred_p   = run_dir / "llm_output.context.json"
    if not (paras_p.exists() and bundle_p.exists() and pred_p.exists()):
        return [0,0], [], [0,0], 0

    paragraphs = json.loads(paras_p.read_text(encoding="utf-8"))
    bundle     = json.loads(bundle_p.read_text(encoding="utf-8"))
    pred_obj   = json.loads(pred_p.read_text(encoding="utf-8"))
    M = len(paragraphs)

                         
    all_headings = [i for i,p in enumerate(paragraphs) if (p.get("level","P")).upper().startswith("H")]

                              
    titles = []
    for it in bundle.get("items", []):
        heads = ((it.get("source_meta") or {}).get("structure") or {}).get("headings") or []
        if heads: titles.append(str(heads[0]).strip())
        else:     titles.append(str(it.get("title") or it.get("topic") or "").strip())

                                                                     
    heading_text = {i: _norm((paragraphs[i].get("text") or "")) for i in all_headings}
    item_starts = []
    last_idx = -1
    for t in titles:
        nt = _norm(t)
        found = None
                                                                                   
        for i in all_headings:
            if i <= last_idx: 
                continue
            htxt = heading_text[i]
            if not htxt:
                continue
            if nt and (nt in htxt or htxt in nt):
                found = i
                break
                                                     
        if found is None:
            for i in all_headings:
                if i > last_idx:
                    found = i
                    break
        if found is not None:
            item_starts.append(found)
            last_idx = found

    gold = [0] + sorted(set(i for i in item_starts if 0 < i < M)) + [M]
    if not gold:
        gold = [0, M]

                      
    pred = pred_obj.get("boundaries") or []
    pred = [0] + sorted(set(int(x) for x in pred if 0 < int(x) < M)) + [M]

    return gold, all_headings, pred, M

def _plot_ribbon(ax, gold_items, all_headings, pred, M):
    if M <= 1:
        ax.axis("off"); return
            
    ax.hlines(0.80, 0, M, colors=(1,0,0,0.20), linewidth=8)              
    ax.hlines(0.20, 0, M, colors=(0,0,1,0.20), linewidth=8)              

                                       
    for h in all_headings:
        ax.vlines(h, 0.70, 0.90, colors="0.6", linewidth=0.8, alpha=0.8)

                                            
    for g in gold_items:
        ax.vlines(g, 0.68, 0.92, colors="red", linewidth=2.0)

                              
    for p in pred:
        ax.vlines(p, 0.08, 0.32, colors="blue", linewidth=2.0)

    ax.set_xlim(0, M); ax.set_ylim(0, 1)
    ax.set_yticks([0.20, 0.80]); ax.set_yticklabels(["pred", "gold (item starts)"])
    ax.set_xlabel("Paragraph index")
    ax.grid(axis="x", alpha=0.15)

def _make_boundary_ribbon_plots(testrun_dir: Path, rows) -> None:
    """
    Creates:
      - plots/ribbons/run_<idx>.png for every run
      - plots/boundary_ribbons_grid.png for a subset (worst F1 by default)
    """
    plots_dir = testrun_dir / "plots"
    rib_dir   = plots_dir / "ribbons"
    rib_dir.mkdir(parents=True, exist_ok=True)

                        
    for r in rows:
        run_dir = testrun_dir / f"run_{r.run_index}"
        gold, all_headings, pred, M = _load_bounds_for_run(run_dir)
        fig, ax = plt.subplots(figsize=(8, 1.6))
        _plot_ribbon(ax, gold, all_headings, pred, M)
        ax.set_title(f"Run {r.run_index} — M={M}, items={r.items_total}, produced={r.produced_chunks}, F1={r.f1:.2f}")
        plt.tight_layout()
        fig.savefig(rib_dir / f"run_{r.run_index}.png", dpi=150)
        plt.close(fig)

                                      
                                                              
    K = min(8, len(rows))
    subset = sorted(rows, key=lambda x: x.f1)[:K]
    if subset:
        n = len(subset)
        fig, axes = plt.subplots(n, 1, figsize=(10, 1.6*n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, r in zip(axes, subset):
            run_dir = testrun_dir / f"run_{r.run_index}"
            gold, all_headings, pred, M = _load_bounds_for_run(run_dir)
            _plot_ribbon(ax, gold, all_headings, pred, M)
            ax.set_title(f"Run {r.run_index} — M={M}, items={r.items_total}, produced={r.produced_chunks}, F1={r.f1:.2f}")
        plt.tight_layout(h_pad=0.6)
        fig.savefig(plots_dir / "boundary_ribbons_grid.png", dpi=150)
        plt.close(fig)

def _save_plots(testrun_dir: Path, rows: List[RunRow], offsets_long: List[Dict[str, Any]]) -> None:
    plots = (testrun_dir / "plots")
    plots.mkdir(parents=True, exist_ok=True)
    if not rows: 
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

                        
    def _scatter(ax, x, y, xlabel, ylabel, diag=False):
        ax.scatter(x, y)
        if diag:
            mn = min(min(x), min(y)); mx = max(max(x), max(y))
            ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    def _box_by_group(ax, groups, values, xlabel, ylabel):
        buckets = defaultdict(list)
        for g, v in zip(groups, values):
            buckets[g].append(v)
        labels = sorted(buckets.keys())
        data = [buckets[k] for k in labels]
        ax.boxplot(data, labels=labels)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    def _hist(ax, vals, bins=30):
        ax.hist(vals, bins=bins)

    def _cdf(ax, vals, xlabel):
        xs = sorted(vals)
        ys = np.linspace(0, 1, num=len(xs), endpoint=True)
        ax.plot(xs, ys)
        ax.set_xlabel(xlabel); ax.set_ylabel("Cumulative share")

    def _stacked_size_bars(ax, inwin, small, big):
        idx = np.arange(len(inwin))
        w = 0.9
        ax.bar(idx, inwin, width=w, label="in-window")
        ax.bar(idx, small, bottom=inwin, width=w, label="too small")
        ax.bar(idx, big, bottom=(np.array(inwin) + np.array(small)), width=w, label="too big")
        ax.set_xlabel("Run index"); ax.set_ylabel("Chunk count")
        ax.set_title("Chunk Size Window per Run"); ax.legend()

                                     
    pages  = [r.pages_total for r in rows]
    items  = [r.items_total for r in rows]
    f1s    = [r.f1 for r in rows]
    precs  = [r.precision for r in rows]
    recs   = [r.recall for r in rows]
    scount = [r.subpdf_count for r in rows]
    topic  = [r.topic_primary for r in rows]
    o50    = [r.offset_p50 for r in rows]
    o90    = [r.offset_p90 for r in rows]

                              
    fig, ax = plt.subplots(); _scatter(ax, pages, f1s, "Pages (total/run)", "F1"); plt.tight_layout(); fig.savefig(plots/"f1_vs_pages.png"); plt.close(fig)
    fig, ax = plt.subplots(); _scatter(ax, items, f1s, "Items (topics/run)", "F1"); plt.tight_layout(); fig.savefig(plots/"f1_vs_items.png"); plt.close(fig)
    fig, ax = plt.subplots(); _scatter(ax, pages, precs, "Pages", "Precision"); plt.tight_layout(); fig.savefig(plots/"precision_vs_pages.png"); plt.close(fig)
    fig, ax = plt.subplots(); _scatter(ax, pages, recs, "Pages", "Recall"); plt.tight_layout(); fig.savefig(plots/"recall_vs_pages.png"); plt.close(fig)
    fig, ax = plt.subplots(); _box_by_group(ax, scount, f1s, "Sub-PDF count", "F1"); plt.tight_layout(); fig.savefig(plots/"f1_by_pdfcount_box.png"); plt.close(fig)

                                     
    agg = defaultdict(list)
    for t, v in zip(topic, f1s):
        agg[t].append(v)
    labels = sorted(agg.keys())
    means  = [float(np.mean(agg[l])) for l in labels]
    stds   = [float(np.std(agg[l])) for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(labels)), 4.5))
    ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=5)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("F1"); ax.set_title("Mean F1 by Primary Topic (±1 SD)")
    plt.tight_layout(); fig.savefig(plots/"f1_by_topic_bar.png"); plt.close(fig)

                                  
    abs_offsets_all = [int(row["abs_offset"]) for row in offsets_long] if offsets_long else []
    if abs_offsets_all:
        fig, ax = plt.subplots(); _hist(ax, abs_offsets_all, bins=range(0, max(abs_offsets_all)+2, 2)); plt.tight_layout(); fig.savefig(plots/"offsets_hist.png"); plt.close(fig)
        fig, ax = plt.subplots(); _cdf(ax, abs_offsets_all, xlabel="|offset| (paras)"); plt.tight_layout(); fig.savefig(plots/"offsets_cdf.png"); plt.close(fig)

                         
    inwin = [max(0, r.produced_chunks - (r.too_small + r.too_big)) for r in rows]
    small = [r.too_small for r in rows]
    big   = [r.too_big for r in rows]
    fig, ax = plt.subplots(figsize=(10,4)); _stacked_size_bars(ax, inwin, small, big); plt.tight_layout(); fig.savefig(plots/"size_window_stacked.png"); plt.close(fig)

                                                         

                                                            
    per = defaultdict(lambda: {"single": [], "mixed": []})
    for r in rows:
        key = r.topic_primary or "unknown"
        group = "single" if r.topic_diversity <= 1 else "mixed"
        per[key][group].append(r.f1)
    labels = sorted(per.keys())
    singles = [np.mean(per[l]["single"]) if per[l]["single"] else 0.0 for l in labels]
    mixed   = [np.mean(per[l]["mixed"]) if per[l]["mixed"] else 0.0 for l in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.8*len(labels)), 4))
    ax.bar(x - w/2, singles, width=w, label="single-topic runs")
    ax.bar(x + w/2, mixed, width=w, label="mixed-topic runs")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("F1")
    ax.set_title("F1 per Primary Topic: Single vs Mixed"); ax.legend()
    plt.tight_layout(); fig.savefig(plots/"topic_perf_mixed_vs_single.png"); plt.close(fig)

                                                                              
                                                           
    mixes = []
    for r in rows:
        run_dir = next((p for p in testrun_dir.glob(f"run_{r.run_index}") if p.is_dir()), None)
        mix = {}
        if run_dir and (run_dir/"combined_bundle.json").exists():
            b = json.loads((run_dir/"combined_bundle.json").read_text(encoding="utf-8"))
            for it in b.get("items", []):
                t = it.get("topic") or it.get("category") or "unknown"
                mix[t] = mix.get(t, 0) + 1
        mixes.append(mix)

    all_topics = sorted({t for m in mixes for t in m.keys()})
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(all_topics)}
    colors["unknown"] = (0.7, 0.7, 0.7, 1.0)

    idx = np.arange(len(rows)); bar_w = 0.35
    plt.figure(figsize=(max(10, len(rows) * 0.9), 5))
    bottoms = np.zeros(len(rows))
    for t in all_topics:
        heights = np.array([mix.get(t, 0) for mix in mixes], dtype=float)
        if heights.sum() == 0: 
            continue
        plt.bar(idx - bar_w/2, heights, bottom=bottoms, width=bar_w, color=colors[t], label=t)
        bottoms += heights

    prod = [r.produced_chunks for r in rows]
    plt.bar(idx + bar_w/2, prod, width=bar_w, color="black", alpha=0.6, label="produced chunks")
    plt.xticks(idx, [r.run_index for r in rows])
    plt.xlabel("Run index"); plt.ylabel("Count")
    plt.title("Boundaries per Run: ground truth (stacked by topic) vs produced")
    handles, labels_ = plt.gca().get_legend_handles_labels()
    if len(handles) > 15:                         
        handles, labels_ = handles[-1:], labels_[-1:]                          
    plt.legend(handles, labels_, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout(); plt.savefig(plots/"boundaries_counts_per_run.png", dpi=150, bbox_inches="tight"); plt.close()

                                          
    fig, ax = plt.subplots(figsize=(6,4))
    _scatter(ax, [r.items_total for r in rows], [r.produced_chunks for r in rows],
             "Ground-truth topics (items_total)", "Produced chunks", diag=True)
    plt.tight_layout(); fig.savefig(plots/"items_vs_chunks_scatter.png"); plt.close(fig)

                                 
    prec = [r.precision for r in rows]; rec = [r.recall for r in rows]; f1 = [r.f1 for r in rows]
    means = [np.mean(prec), np.mean(rec), np.mean(f1)]
    stds  = [np.std(prec),  np.std(rec),  np.std(f1)]
    labels = ["Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(np.arange(3), means, yerr=stds, capsize=5)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Score")
    ax.set_title("Overall PRF1 (mean ± SD)")
    plt.tight_layout(); plt.savefig(plots/"prf1_overview.png"); plt.close(fig)
    _make_boundary_ribbon_plots(testrun_dir, rows)

                                                          
async def _auto_test_worker(job_id: str, cfg: StartPayload):
    rnd = random.Random(cfg.seed if cfg.seed is not None else (int(datetime.now().timestamp() * 1000) & 0xFFFFFFFF))
    suffix = _batch_suffix(cfg)
    ts = _ts_dir()
    name = f"testrun_{ts}" if not suffix else f"testrun_{ts}_{suffix}"
    testrun_dir = BASE_OUTDIR / name
    testrun_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = cfg.model_dump()
    mods_cfg = _extract_mods(cfg)

                                            
    runs_total = int(cfg.runs)
    using_cache = bool(cfg.use_cached)
    cached_runs: List[Path] = []
    cache_manifest: Dict[str, Any] = {}
    if using_cache:
        cache_root = _cache_root(cfg.use_cached)
        def _cache_sort_key(path: Path) -> tuple:
            suffix = path.name.split("_")[-1]
            idx = int(suffix) if suffix.isdigit() else float("inf")
            return (idx, path.name)
        cached_runs = sorted(
            [p for p in cache_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
            key=_cache_sort_key,
        )
        if not cached_runs:
            raise RuntimeError(f"No runs found in cached dataset '{cfg.use_cached}'")
        runs_total = len(cached_runs)
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists():
            try:
                cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                cache_manifest = {}
    if using_cache:
        plan_docs = cache_manifest.get("docs_per_run") or [0] * runs_total
        if len(plan_docs) != runs_total:
            plan_docs = [0] * runs_total
        else:
            try:
                plan_docs = [int(x) for x in plan_docs]
            except Exception:
                plan_docs = [0] * runs_total
    else:
        plan_docs = [rnd.randint(int(cfg.min_pdfs), int(cfg.max_pdfs)) for _ in range(runs_total)]
    docs_total = sum(plan_docs) if any(plan_docs) else runs_total
    _write_testrun_meta(testrun_dir, cfg, runs_total, plan_docs, mods_cfg)

    async with JOBS_LOCK:
        JOBS[job_id].update({
            "status": "running",
            "testrun_dir": str(testrun_dir.resolve()),
            "docs_total": int(docs_total),
            "docs_done": 0,
            "run_started": 0,
            "run_finished": 0,
            "phase": "generating",
            "current": "Starting…",
            "eta_seconds": None,
            "started_at": time.time(),
            "total": runs_total,
        })

    try:
                                                                                     
        for i in range(runs_total):
            run_dir = testrun_dir / f"run_{i}"
            if using_cache:
                src = cached_runs[i]
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                async with JOBS_LOCK:
                    JOBS[job_id]["current"] = f"run {i+1}/{runs_total} — loading cache '{cfg.use_cached}'"
                    JOBS[job_id]["phase"] = "loading-cache"
                await asyncio.to_thread(shutil.copytree, src, run_dir)
                inc = plan_docs[i] if plan_docs else 1
                if inc <= 0:
                    inc = 1
                async with JOBS_LOCK:
                    JOBS[job_id]["docs_done"] = min(
                        JOBS[job_id].get("docs_total", 0),
                        JOBS[job_id].get("docs_done", 0) + inc
                    )
                    done = JOBS[job_id]["docs_done"]
                    total = JOBS[job_id]["docs_total"] or runs_total
                    elapsed = time.time() - JOBS[job_id]["started_at"]
                    rate = (done / elapsed) if elapsed > 1e-6 else 0.0
                    eta = (total - done) / rate if rate > 0 else None
                    JOBS[job_id]["eta_seconds"] = float(eta) if eta else None
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            num_pdfs = max(1, int(plan_docs[i]))

            for j in range(num_pdfs):
                pages = rnd.randint(int(cfg.min_pages), int(cfg.max_pages))
                async with JOBS_LOCK:
                    JOBS[job_id]["current"] = f"run {i+1}/{runs_total} — generating PDF {j+1}/{num_pdfs} (~{pages} pages)"
                    JOBS[job_id]["phase"] = "generating"
                lengths_arg = None if (cfg.length_mode in (None, "mixed")) else [cfg.length_mode]
                                     
                await asyncio.to_thread(
                    create_test_data,
                    pages,
                    run_dir,
                    source_dir=SOURCE_DIR,
                    seed=cfg.seed,
                    selection_mode=("topics" if (cfg.topics and len(cfg.topics)>0) else (cfg.selection_mode or "mixed")),
                    lengths=lengths_arg,
                )
                async with JOBS_LOCK:
                    JOBS[job_id]["docs_done"] += 1
                    done = JOBS[job_id]["docs_done"]
                    total = JOBS[job_id]["docs_total"]
                    elapsed = time.time() - JOBS[job_id]["started_at"]
                    rate = (done / elapsed) if elapsed > 1e-6 else 0.0
                    eta = (total - done) / rate if rate > 0 else None
                    JOBS[job_id]["eta_seconds"] = float(eta) if eta else None

            if cfg.cache_name:
                async with JOBS_LOCK:
                    JOBS[job_id]["current"] = f"run {i+1}/{runs_total} - caching PDFs to '{cfg.cache_name}'"
                await asyncio.to_thread(_cache_run_artifacts, run_dir, cfg.cache_name, i)

        if cfg.cache_name:
            await asyncio.to_thread(_write_cache_manifest, cfg.cache_name, cfg_dump, plan_docs, testrun_dir)

        if cfg.cache_only:
            async with JOBS_LOCK:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["phase"] = "cached"
                JOBS[job_id]["current"] = f"Cached dataset '{cfg.cache_name}'"
                JOBS[job_id]["docs_done"] = JOBS[job_id].get("docs_total", 0)
                JOBS[job_id]["completed"] = runs_total
                JOBS[job_id]["eta_seconds"] = 0.0
            return

                                                                      
        async with JOBS_LOCK:
            JOBS[job_id]["phase"] = "processing"
            JOBS[job_id]["current"] = "Chunking & evaluating…"

        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        rows: List[RunRow] = []
        rows_lock = asyncio.Lock()
        offsets_long_rows: List[Dict[str, Any]] = []

        async def do_run(i: int):
            run_dir = testrun_dir / f"run_{i}"
            async with sem:
                async with JOBS_LOCK:
                    JOBS[job_id]["run_started"] += 1
                    JOBS[job_id]["current"] = f"processing run {i+1}/{runs_total}"
                                      
                row: RunRow = await asyncio.to_thread(_process_and_eval_run, run_dir, mods_cfg)
                                 
                off_csv = run_dir / "offsets.csv"
                local_offsets: List[Dict[str, Any]] = []
                if off_csv.exists():
                    with off_csv.open("r", encoding="utf-8") as f:
                        _ = f.readline()
                        for ln in f:
                            off, absoff, topic, spdf, pgs, items = ln.strip().split(",")
                            local_offsets.append({
                                "run_index": row.run_index,
                                "offset": int(off),
                                "abs_offset": int(absoff),
                                "primary_topic": topic,
                                "subpdf_count": int(spdf),
                                "pages_total": int(pgs),
                                "items_total": int(items),
                            })
                async with rows_lock:
                    rows.append(row)
                    offsets_long_rows.extend(local_offsets)
                async with JOBS_LOCK:
                    JOBS[job_id]["run_finished"] += 1

        tasks = [asyncio.create_task(do_run(i)) for i in range(runs_total)]
        await asyncio.gather(*tasks)

                                                     
        async with JOBS_LOCK:
            JOBS[job_id]["phase"] = "summarizing"
            JOBS[job_id]["current"] = "Writing summaries & plots…"

        if rows:
            sumcsv = testrun_dir / "summary.csv"
            with sumcsv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "run_index","subpdf_count","pages_total","items_total",
                    "desired_chunks","produced_chunks","paragraphs",
                    "chunk_diff","chunk_diff_pct","chunk_size_diff_avg","chunk_size_diff_pct",
                    "chunk_time_sec","mods_used",
                    "precision","recall","f1","mabo",
                    "too_small","too_big","strictly_increasing",
                    "first_is_zero","last_is_M","duplicate_count","report_dir"
                ])
                w.writeheader()
                for r in rows:
                    w.writerow({
                        "run_index": r.run_index,
                        "subpdf_count": r.subpdf_count,
                        "pages_total": r.pages_total,
                        "items_total": r.items_total,
                        "desired_chunks": r.desired_chunks,
                        "produced_chunks": r.produced_chunks,
                        "paragraphs": r.paragraphs,
                        "chunk_diff": r.chunk_diff,
                        "chunk_diff_pct": f"{r.chunk_diff_pct:.4f}",
                        "chunk_size_diff_avg": f"{r.chunk_size_diff_avg:.4f}",
                        "chunk_size_diff_pct": f"{r.chunk_size_diff_pct:.4f}",
                        "chunk_time_sec": f"{r.chunk_time_sec:.2f}",
                        "mods_used": r.mods_used,
                        "precision": f"{r.precision:.4f}",
                        "recall": f"{r.recall:.4f}",
                        "f1": f"{r.f1:.4f}",
                        "mabo": f"{r.mabo:.4f}",
                        "too_small": r.too_small,
                        "too_big": r.too_big,
                        "strictly_increasing": r.strictly_increasing,
                        "first_is_zero": r.first_is_zero,
                        "last_is_M": r.last_is_M,
                        "duplicate_count": r.duplicate_count,
                        "report_dir": r.report_dir
                    })

            suffix = _batch_suffix(cfg)
            sumall_name = f"summary_all_{suffix}.csv" if suffix else "summary_all.csv"
            sumall = testrun_dir / sumall_name
            with sumall.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(asdict(r))

            if offsets_long_rows:
                offall = testrun_dir / "testrun_offsets.csv"
                with offall.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(offsets_long_rows[0].keys()))
                    w.writeheader()
                    for row in offsets_long_rows:
                        w.writerow(row)

            await asyncio.to_thread(_save_plots, testrun_dir, rows, offsets_long_rows)

        async with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["current"] = "All runs completed"
            JOBS[job_id]["eta_seconds"] = 0.0

    except Exception as e:
        async with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["current"] = "Error"
