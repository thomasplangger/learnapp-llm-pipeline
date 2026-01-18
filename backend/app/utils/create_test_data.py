# backend/app/utils/create_test_data.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from app.utils.stitch_texts_to_pdf import run_stitch

def create_test_data(
    total_pages: int,
    outdir: Union[str, Path],
    *,
    source_dir: Union[str, Path],
    selection_mode: str = "mixed",
    topics: Optional[List[str]] = None,
    lengths: Optional[List[str]] = None,
    pagesize: str = "A4",
    page_margins_cm: float = 2.0,
    font_ttf: Optional[str] = None,
    approx_words_per_token: float = 0.75,
    approx_words_per_page: int = 550,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper that delegates to run_stitch() and returns the bundle.
    Keeps the same function name your router calls.
    """
    lengths = lengths or ["short", "medium", "long"]

    bundle = run_stitch(
        indir=Path(source_dir),
        outdir=Path(outdir),
        target_metric="total_pages",
        target_value=int(total_pages),
        topics=(topics or None),
        lengths=lengths,
        approx_words_per_token=approx_words_per_token,
        approx_words_per_page=approx_words_per_page,
        seed=seed,
        page_margins_cm=page_margins_cm,
        pagesize=pagesize,
        preview_html=False,
        font_ttf=font_ttf,
        selection_mode=selection_mode,
    )

    out_dir = Path(bundle["inputs"]["out_run_dir"])
    return {
        **bundle,
        "run_dir": str(out_dir),
        "pdf": str(out_dir / bundle["files"]["combined_pdf"]),
        "json": str(out_dir / Path(bundle["files"]["combined_pdf"]).with_suffix(".json").name),
        "csv": str(out_dir / bundle["files"]["summary_csv"]),
        "pages_measured": bundle["totals"]["pages_measured"],
        "count": bundle["selection"]["count"],
    }
