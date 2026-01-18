# app/utils/markdown_to_json.py
from __future__ import annotations
import re
from typing import List, Dict, Any

_LIST_MARKER_RX = re.compile(r"""^\s*(
    [\-\*\+•·‣]            # bullets
  | \d+[\.\)]              # 1. or 1)
  | [a-zA-Z][\.\)]         # a. or a)
  | \(\d+\)                # (1)
  | \([a-zA-Z]\)           # (a)
)\s+""", re.VERBOSE)

_HDR_MD_RX = re.compile(r'^(#{1,6})\s+(.*)$')

def is_list_line(text: str) -> bool:
    return bool(_LIST_MARKER_RX.match(text or ""))

def normalize_whitespace(s: str) -> str:
    return re.sub(r'[ \t]+', ' ', (s or '')).strip()

def finalize_list_runs(items: List[Dict[str, Any]]) -> None:
    """
    Assign list_index to consecutive list runs of length >= 3.
    Shorter runs are downgraded to non-list paragraphs.
    """
    current_index = 0
    i = 0
    n = len(items)
    while i < n:
        if items[i].get("is_list"):
            j = i
            while j < n and items[j].get("is_list"):
                j += 1
            run_len = j - i
            if run_len >= 3:
                for k in range(i, j):
                    items[k]["list_index"] = current_index
                current_index += 1
            else:
                for k in range(i, j):
                    items[k]["is_list"] = False
                    items[k]["list_index"] = None
            i = j
        else:
            i += 1

def assign_paths(items: List[Dict[str, Any]]) -> None:
    """
    Maintain a stack of heading indices; non-heading items inherit current stack.
    Heading depths: H1->1, H2->2, H3->3. Anything else -> depth 0.
    """
    def depth(level: str) -> int:
        return {"H1": 1, "H2": 2, "H3": 3}.get(level, 0)

    stack: List[int] = []
    for it in items:
        d = depth(it["level"])
        if d > 0:
            while len(stack) >= d:
                stack.pop()
            stack.append(it["i"])
            it["path"] = stack[:-1]
        else:
            it["path"] = list(stack)


def parse_docling_md(md_text: str) -> List[Dict[str, Any]]:
    """
    Parse a Markdown string into normalized paragraph objects:
    { i, level: H1|H2|H3|P, is_list, list_index, path, text }
    """
    lines = [ln.rstrip() for ln in (md_text or "").splitlines()]
    out: List[Dict[str, Any]] = []
    buf: List[str] = []
    idx = 0

    def flush_paragraph():
        nonlocal idx
        if not buf:
            return
        text = normalize_whitespace(" ".join(buf))
        if text:
            out.append({
                "i": idx, "level": "P", "is_list": is_list_line(text),
                "list_index": None, "path": [], "text": text
            })
            idx += 1
        buf.clear()

    for ln in lines:
        if not ln.strip():
            flush_paragraph()
            continue

        m = _HDR_MD_RX.match(ln)
        if m:
            flush_paragraph()
            hashes, content = m.group(1), m.group(2).strip()
            level = {1: "H1", 2: "H2", 3: "H3"}.get(min(len(hashes), 3), "H3")
            out.append({
                "i": idx, "level": level, "is_list": False,
                "list_index": None, "path": [], "text": content
            })
            idx += 1
        else:
            if is_list_line(ln):
                flush_paragraph()
                out.append({
                    "i": idx, "level": "P", "is_list": True,
                    "list_index": None, "path": [], "text": normalize_whitespace(ln)
                })
                idx += 1
            else:
                buf.append(ln)

    flush_paragraph()
    finalize_list_runs(out)
    assign_paths(out)
    return out

def markdown_to_structured(md_text: str) -> List[Dict[str, Any]]:
    return parse_docling_md(md_text)

def parse_docling_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Heuristic flattener for Docling JSON into normalized paragraph objects.
    Traverses body.children (refs to texts/groups), resolves #/texts/* in order,
    maps headings to H1/H2/H3 when possible, otherwise P. Applies same list/path rules.
    """
    texts = doc.get("texts", [])
    groups = doc.get("groups", [])
    by_ref: Dict[str, Any] = {}
    for name, arr in (("texts", texts), ("groups", groups)):
        for i, node in enumerate(arr):
            by_ref[f"#/{name}/{i}"] = node

    def resolve_text(node: Dict[str, Any]) -> str:
        for k in ("value", "text", "content", "string"):
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v
        spans = node.get("spans")
        if isinstance(spans, list):
            s = " ".join([normalize_whitespace(str(s.get("text", ""))) for s in spans if isinstance(s, dict)])
            if s.strip():
                return s
        return ""

    def node_level(node: Dict[str, Any]) -> str:
        label = (node.get("label") or node.get("role") or "").lower()
        name = (node.get("name") or "").lower()
        for val in (label, name):
            if "heading" in val or val.startswith("h1"):
                return "H1"
            if val.startswith("h2"):
                return "H2"
            if val.startswith("h3"):
                return "H3"
            if val in ("title", "section", "chapter"):
                return "H1"
        t = resolve_text(node)
        if t and len(t) <= 80 and (t == t.upper() or t.istitle()):
            return "H3"
        return "P"

    ordered_nodes: List[Dict[str, Any]] = []
    body = doc.get("body", {})
    for ch in body.get("children", []):
        if isinstance(ch, dict) and "$ref" in ch and ch["$ref"] in by_ref:
            ordered_nodes.append(by_ref[ch["$ref"]])
        elif isinstance(ch, dict) and "children" in ch:
            for sub in ch.get("children", []):
                if isinstance(sub, dict) and "$ref" in sub and sub["$ref"] in by_ref:
                    ordered_nodes.append(by_ref[sub["$ref"]])

    if not ordered_nodes and isinstance(texts, list):
        ordered_nodes = list(texts)

    out: List[Dict[str, Any]] = []
    idx = 0
    for node in ordered_nodes:
        text = resolve_text(node)
        if not text.strip():
            continue
        level = node_level(node)
        islist = is_list_line(text)
        out.append({
            "i": idx, "level": level, "is_list": islist,
            "list_index": None, "path": [], "text": normalize_whitespace(text)
        })
        idx += 1

    finalize_list_runs(out)
    assign_paths(out)
    return out
