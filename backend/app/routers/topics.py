# app/routers/topics.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["topics"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _safe_resolve(base: str) -> Path:
    """
    Resolve 'base' safely relative to PROJECT_ROOT.

    Accepted examples:
      - "generated"              -> <PROJECT_ROOT>/generated
      - "backend/generated"      -> <PROJECT_ROOT>/generated
      - absolute path            -> used as-is (must be under PROJECT_ROOT)
    """
    p = Path(base)

    if not p.is_absolute():
        parts = p.parts
        if parts and parts[0].lower() == "backend":
            p = PROJECT_ROOT.joinpath(*parts[1:])
        else:
            p = PROJECT_ROOT / p
    else:
        p = p.resolve()

    try:
        p.relative_to(PROJECT_ROOT)
    except Exception:
        raise HTTPException(status_code=400, detail="Base path escapes project root.")

    return p

def _folder_stats(path: Path) -> Dict[str, Any]:
    pdfs = list(path.glob("**/*.pdf"))
    jsons = list(path.glob("**/*.json"))
    csvs = list(path.glob("**/*.csv"))
    latest_mtime = 0.0
    for fp in pdfs + jsons + csvs:
        try:
            latest_mtime = max(latest_mtime, fp.stat().st_mtime)
        except Exception:
            pass
    return {
        "name": path.name,
        "path": str(path),
        "pdf_count": len(pdfs),
        "json_count": len(jsons),
        "csv_count": len(csvs),
        "updated_at": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime else None,
    }

@router.get("/topics")
def list_topics(base: str = Query("backend/generated", description="Base directory to scan for topics")) -> List[Dict[str, Any]]:
    base_path = _safe_resolve(base)
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Base path not found: {base_path}")
    if not base_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Base path is not a directory: {base_path}")

    topics: List[Dict[str, Any]] = []
    for entry in sorted(base_path.iterdir()):
        if entry.is_dir():
            topics.append(_folder_stats(entry))
    return topics
