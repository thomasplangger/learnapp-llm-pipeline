# app/utils/pdf_to_markdown.py
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

from bson import ObjectId
from fastapi import HTTPException

from app.db import fs

def _docling_markdown_from_bytes(raw: bytes) -> Optional[str]:
    """
    Try Docling -> Markdown from raw PDF bytes using the stable DocumentConverter API.
    Returns Markdown on success, else None.
    This path is version-resilient across Docling releases.
    """
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return None

    try:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "input.pdf"
            pdf_path.write_bytes(raw)
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            doc = getattr(result, "document", None) or getattr(result, "output_document", None)
            if doc is None:
                return None
            for meth in ("export_to_markdown", "export_markdown", "to_markdown"):
                fn = getattr(doc, meth, None)
                if callable(fn):
                    md = fn()
                    if isinstance(md, str) and md.strip():
                        return md.strip()
    except Exception:
        return None
    return None


def _pymupdf_markdown_from_bytes(raw: bytes) -> str:
    """
    Fallback using PyMuPDF. Uses 'markdown' if available, else 'text'.
    """
    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "Neither Docling nor PyMuPDF produced markdown. "
            "Install one of: `pip install docling` or `pip install pymupdf`."
        ) from e

    doc = fitz.open(stream=raw, filetype="pdf")
    out_parts = []
    for page in doc:
        try:
            md = page.get_text("markdown")
        except Exception:
            md = page.get_text("text")
        out_parts.append((md or "").strip())
        out_parts.append("\n\n")
    return "".join(out_parts).strip()


def pdf_bytes_to_markdown(pdf_bytes: bytes, *, fast: bool = True, device: str | None = None) -> str:
    """
    Convert PDF bytes to Markdown.
    - First tries Docling's DocumentConverter (version-stable path).
    - Falls back to PyMuPDF if Docling not present or fails.

    Notes:
      * The `fast` and `device` knobs are placeholders kept for API compatibility.
        They matter only if you later re-introduce a Docling pipeline that exposes such options.
    """
    md = _docling_markdown_from_bytes(pdf_bytes)
    if isinstance(md, str) and md.strip():
        return md
    return _pymupdf_markdown_from_bytes(pdf_bytes)

async def _load_pdf_bytes(pdf_id: str) -> bytes:
    try:
        oid = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pdf_id")

    try:
        stream = await fs.open_download_stream(oid)
        raw = await stream.read()
        if not raw:
            raise HTTPException(status_code=404, detail="PDF is empty")
        return raw
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="PDF not found or unreadable")


async def get_pdf_markdown(pdf_id: str) -> str:
    """
    Download the PDF bytes from GridFS by ObjectId and return Markdown
    (Docling preferred, PyMuPDF fallback).
    """
    raw = await _load_pdf_bytes(pdf_id)
    md = pdf_bytes_to_markdown(raw)
    if not md.strip():
        raise HTTPException(status_code=422, detail="Unable to extract Markdown from PDF")
    return md


async def get_combined_markdown_and_offsets(
    pdf_ids: List[str]
) -> Tuple[str, List[Tuple[int, int, str, int]]]:
    """
    Return (combined_markdown, source_map) for multiple PDFs.

    - Each PDF is converted to Markdown independently.
    - PDFs are joined with a clear Markdown separator that preserves paragraph boundaries.
    - source_map: list of (start, end, pdf_id, pdf_index) spans within the combined buffer.
      (end is exclusive)
    """
    parts: List[str] = []
    source_map: List[Tuple[int, int, str, int]] = []
    pos = 0

    def sep_for(idx: int, pid: str) -> str:
        return f"\n\n---\n\n<!-- PDF {idx+1}: {pid} -->\n\n"

    for idx, pid in enumerate(pdf_ids):
        if idx > 0:
            s = sep_for(idx, pid)
            parts.append(s)
            pos += len(s)

        try:
            md = await get_pdf_markdown(pid)
        except Exception:
            md = ""
        start = pos
        if md:
            parts.append(md)
            pos += len(md)
        end = pos
        source_map.append((start, end, pid, idx))

    combined = "".join(parts)
    return combined, source_map