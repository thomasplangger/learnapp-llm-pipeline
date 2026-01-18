import json, math, re, os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

BULLET_RE = re.compile(r'^(?:\s*(?:[-*•]\s+|\d+\.\s+))')

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def safe_first_line(s: str, n: int = 120) -> str:
    if not s: return ""
    s = s.strip().splitlines()[0] if "\n" in s else s.strip()
    return (s[:n] + " …") if len(s) > n else s

def est_tokens_from_text(s: str) -> int:
    return max(1, len(s) // 4)

def chunk_sizes(paragraphs: List[Dict], bounds: List[int]) -> Tuple[List[int], List[int]]:
    para_counts, tok_counts = [], []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i+1]
        para_counts.append(b - a)
        toks = 0
        for p in paragraphs[a:b]:
            if "tok" in p: toks += int(p.get("tok", 0))
            else: toks += est_tokens_from_text(p.get("text", ""))
        tok_counts.append(toks)
    return para_counts, tok_counts

def coef_variation(vals: List[float]) -> float:
    if not vals: return 0.0
    m = sum(vals) / len(vals)
    if m == 0: return 0.0
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(var) / m

def gini(vals: List[float]) -> float:
    if not vals: return 0.0
    xs = sorted(vals)
    n = len(xs)
    cum = 0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n

def is_header(level: str) -> bool:
    return level in ("H1", "H2", "H3")

def boundary_quality(paragraphs: List[Dict], bounds: List[int]) -> Dict[str, Any]:
    total = max(0, len(bounds) - 2)
    header_starts = 0
    header_orphan = 0
    list_splits = 0

    for i in range(1, len(bounds) - 1):
        at = bounds[i]
        if at < len(paragraphs):
            p = paragraphs[at]
            if is_header(p.get("level", "P")):
                header_starts += 1
            if at > 0 and is_header(paragraphs[at - 1].get("level", "P")):
                header_orphan += 1
            if p.get("is_list") or BULLET_RE.match((p.get("text") or "").lstrip() or ""):
                list_splits += 1

    pct = lambda x: (100.0 * x / total) if total else 0.0
    return {
        "interior_boundaries": total,
        "pct_starts_on_header": pct(header_starts),
        "count_starts_on_header": header_starts,
        "pct_header_orphan": pct(header_orphan),
        "count_header_orphan": header_orphan,
        "pct_list_split": pct(list_splits),
        "count_list_split": list_splits,
    }

def strict_integrity(bounds: List[int], M: int) -> Dict[str, Any]:
    unique_sorted = sorted(set(bounds))
    strictly_increasing = all(unique_sorted[i] < unique_sorted[i+1] for i in range(len(unique_sorted)-1))
    first_ok = (unique_sorted[0] == 0)
    last_ok = (unique_sorted[-1] == M)
    gaps = []
    dups = len(bounds) - len(unique_sorted)
    if not strictly_increasing or not first_ok or not last_ok or dups:
        seen = set()
        for i, b in enumerate(bounds):
            if b in seen: gaps.append(("dup", i, b))
            seen.add(b)
        if not strictly_increasing:
            for i in range(len(bounds)-1):
                if bounds[i] >= bounds[i+1]:
                    gaps.append(("non_increasing", i, bounds[i], bounds[i+1]))
        if not first_ok: gaps.append(("first_not_zero", bounds[0]))
        if not last_ok: gaps.append(("last_not_M", bounds[-1], M))
    return {
        "strictly_increasing": strictly_increasing,
        "first_is_zero": first_ok,
        "last_is_M": last_ok,
        "duplicate_count": dups,
        "issues": gaps,
    }

def computed_size_window(M: int, N: int) -> Tuple[int, int]:
    avg = max(1, round(M / max(1, N)))
    min_p = max(1, math.floor(avg * 0.75))
    max_p = max(min_p + 1, math.ceil(avg * 1.35))
    return min_p, max_p

def size_violations(para_counts: List[int], min_p: int, max_p: int) -> Dict[str, Any]:
    too_small = [i for i, c in enumerate(para_counts) if c < min_p]
    too_big = [i for i, c in enumerate(para_counts) if c > max_p]
    return {"too_small": too_small, "too_big": too_big}

def align_with_reasons(bounds: List[int], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
    at_values = [r.get("at") for r in reasons if isinstance(r.get("at"), int)]
    set_bounds = set(bounds[1:-1])
    matched = [a for a in at_values if a in set_bounds]
    missing = [a for a in at_values if a not in set_bounds]
    extras = [b for b in set_bounds if b not in at_values]
    return {"reasons_total": len(at_values), "matched": len(matched), "missing_reasons_for": missing, "extra_boundaries_without_reasons": extras}

def titles_from_bundle(bundle: Dict[str, Any], *, mode: str = "all") -> List[str]:
    """
    Extract titles used to locate gold boundary paragraph indices.

    mode:
      - "all": return every heading title found in each bundle item (fine-grained, heading-level)
      - "first": return only the first heading per bundle item (coarser, item-level)
    """
    if mode not in {"all", "first"}:
        raise ValueError("mode must be 'all' or 'first'")

    titles: List[str] = []
    for it in bundle.get("items", []):
        meta = it.get("source_meta", {}) or {}
        struct = meta.get("structure", {}) or {}
        heads = struct.get("headings", []) or []

        def _title_from_heading(h: Any) -> str:
            if isinstance(h, dict):
                return (h.get("title") or "").strip()
            return str(h or "").strip()

        if mode == "first":
            if heads:
                titles.append(_title_from_heading(heads[0]))
            else:
                titles.append(str(it.get("title") or it.get("topic") or it.get("category") or "").strip())
        else:
            for h in heads:
                titles.append(_title_from_heading(h))

    return [t for t in titles if t]

def find_paragraph_index_by_title(paragraphs: List[Dict], title: str) -> Optional[int]:
    norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]+', '', s or '')).strip().lower()
    target = norm(title)
    for p in paragraphs:
        if norm(p.get("text", "")) == target:
            return int(p["i"])
    ft = norm(safe_first_line(title))
    for p in paragraphs:
        if norm(safe_first_line(p.get("text",""))) == ft:
            return int(p["i"])
    return None

def boundary_near(a: int, bset: set, tol: int = 1) -> bool:
    return any((a + d) in bset for d in range(-tol, tol+1))

def compare_to_gold(bounds: List[int], gold_bounds: List[int], tol: int = 2) -> Dict[str, Any]:
    B = set(bounds[1:-1])
    G = set(gold_bounds[1:-1])
    tp = sum(1 for g in G if boundary_near(g, B, tol))
    fp = sum(1 for b in B if not boundary_near(b, G, tol))
    fn = sum(1 for g in G if not boundary_near(g, B, tol))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
    matched_offsets = []
    for g in G:
        if boundary_near(g, B, tol):
            offsets = [abs(b - g) for b in B]
            matched_offsets.append(min(offsets) if offsets else 0)
    mabo = sum(matched_offsets)/len(matched_offsets) if matched_offsets else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "mabo": mabo}

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def evaluate(paragraphs_path: str,
             llm_output_path: str,
             bundle_json_path: Optional[str] = None,
             gold_csv_path: Optional[str] = None,
             desired_chunks: Optional[int] = None,
             out_dir: Optional[str] = None,
             size_window_from_desired: bool = True) -> str:

    paragraphs = load_json(Path(paragraphs_path))
    llm = load_json(Path(llm_output_path))
    bundle = load_json(Path(bundle_json_path)) if bundle_json_path else None
    reasons = llm.get("reasons", [])
    bounds = llm.get("boundaries", [])
    M = len(paragraphs)

    b = sorted(set(int(x) for x in bounds if isinstance(x, int)))
    if not b or b[0] != 0: b = [0] + b
    if b[-1] != M: b = b + [M]
    clean = [b[0]]
    for x in b[1:]:
        if x > clean[-1]:
            clean.append(x)
    bounds = clean

    stamp = datetime.now()
    out_base = Path(out_dir or "debu_files") / f"eval_{stamp.year}_{stamp.month:02d}_{stamp.day:02d}_{stamp.hour:02d}_{stamp.minute:02d}"
    out_base.mkdir(parents=True, exist_ok=True)

    integ = strict_integrity(bounds, M)

    para_counts, tok_counts = chunk_sizes(paragraphs, bounds)
    N_chunks = len(bounds) - 1
    min_p, max_p = (computed_size_window(M, desired_chunks or N_chunks)
                    if size_window_from_desired else computed_size_window(M, N_chunks))
    viol = size_violations(para_counts, min_p, max_p)

    size_stats = {
        "chunks": N_chunks,
        "paras_total": M,
        "paras_min": min(para_counts) if para_counts else 0,
        "paras_max": max(para_counts) if para_counts else 0,
        "paras_mean": (sum(para_counts)/N_chunks) if N_chunks else 0,
        "paras_stdev": (math.sqrt(sum((c - (sum(para_counts)/N_chunks))**2 for c in para_counts)/N_chunks) if N_chunks else 0),
        "paras_cv": coef_variation(para_counts),
        "paras_gini": gini(para_counts),
        "tokens_total_est": sum(tok_counts),
        "tokens_min": min(tok_counts) if tok_counts else 0,
        "tokens_max": max(tok_counts) if tok_counts else 0,
        "tokens_mean": (sum(tok_counts)/N_chunks) if N_chunks else 0,
        "tokens_cv": coef_variation(tok_counts),
        "target_window_min_p": min_p,
        "target_window_max_p": max_p,
        "too_small_count": len(viol["too_small"]),
        "too_big_count": len(viol["too_big"]),
    }

    bq = boundary_quality(paragraphs, bounds)
    rz = align_with_reasons(bounds, reasons)

    heading_hits = None
    if bundle:
        titles = titles_from_bundle(bundle)
        idxs = []
        for t in titles:
            idx = find_paragraph_index_by_title(paragraphs, t)
            if idx is not None:
                idxs.append(idx)
        idxs = sorted(set(idxs))
        B = set(bounds[1:-1])
        matches = sum(1 for h in idxs if boundary_near(h, B, 1))
        heading_hits = {
            "bundle_titles_found": len(idxs),
            "boundaries_near_bundle_titles": matches,
            "pct_boundaries_near_titles": (100.0 * matches / max(1, len(B))),
        }

    gold_scores = None
    if gold_csv_path and Path(gold_csv_path).exists():
        gold_lines = Path(gold_csv_path).read_text(encoding="utf-8").strip().splitlines()
        gold = []
        for ln in gold_lines:
            try: gold.append(int(ln.strip()))
            except: pass
        gold = sorted(set(x for x in gold if 0 <= x <= M))
        if not gold or gold[0] != 0: gold = [0] + gold
        if gold[-1] != M: gold.append(M)
        gold_scores = compare_to_gold(bounds, gold, tol=2)
        rows = [{"gold_boundary": x} for x in gold]
        write_csv(out_base / "gold_boundaries.csv", rows)

    chunk_rows = []
    for i in range(N_chunks):
        a, b = bounds[i], bounds[i+1]
        head = paragraphs[a] if a < len(paragraphs) else {}
        chunk_rows.append({
            "chunk": i,
            "start": a,
            "end": b,
            "paras": b - a,
            "tokens_est": tok_counts[i] if i < len(tok_counts) else 0,
            "starts_with_level": head.get("level", ""),
            "starts_with_text": safe_first_line(head.get("text", "")),
        })
    write_csv(out_base / "chunks.csv", chunk_rows)

    b_rows = []
    for i, at in enumerate(bounds):
        lev = paragraphs[at].get("level","") if at < len(paragraphs) else ""
        txt = safe_first_line(paragraphs[at].get("text","")) if at < len(paragraphs) else ""
        b_rows.append({"k": i, "at": at, "level": lev, "first_line": txt})
    write_csv(out_base / "boundaries.csv", b_rows)

    if rz["missing_reasons_for"] or rz["extra_boundaries_without_reasons"]:
        rows = []
        for m in rz["missing_reasons_for"]:
            rows.append({"type": "missing_reason_for_boundary", "index": m})
        for e in rz["extra_boundaries_without_reasons"]:
            rows.append({"type": "boundary_without_reason", "index": e})
        write_csv(out_base / "mismatch_reasons.csv", rows)

    md = []
    md.append(f"# Chunking Evaluation\n")
    md.append(f"- Paragraphs (M): **{M}**")
    md.append(f"- Chunks (N): **{N_chunks}**  {'(requested: ' + str(desired_chunks) + ')' if desired_chunks else ''}\n")
    md.append("## Integrity\n")
    md.append(f"- Strictly increasing: **{integ['strictly_increasing']}**")
    md.append(f"- First boundary is 0: **{integ['first_is_zero']}**")
    md.append(f"- Last boundary is M: **{integ['last_is_M']}**")
    md.append(f"- Duplicate boundaries: **{integ['duplicate_count']}**")
    if integ["issues"]:
        md.append(f"- Issues detail: `{integ['issues']}`")
    md.append("\n## Size & Balance\n")
    for k, v in size_stats.items():
        md.append(f"- {k}: **{v}**")
    md.append("\n## Boundary Quality\n")
    for k, v in bq.items():
        md.append(f"- {k}: **{v}**")
    md.append("\n## Reasons Alignment\n")
    for k, v in rz.items():
        md.append(f"- {k}: **{v}**")
    if heading_hits:
        md.append("\n## Alignment with Bundle Headings\n")
        for k, v in heading_hits.items():
            md.append(f"- {k}: **{v}**")
    if gold_scores:
        md.append("\n## Gold Comparison (±2 paragraphs)\n")
        for k, v in gold_scores.items():
            md.append(f"- {k}: **{v}**")

    (out_base / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[eval] wrote report to: {out_base}")
    return str(out_base)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Evaluate chunking quality")
    ap.add_argument("--paragraphs", required=True, help="debug_* .paragraphs_used.json")
    ap.add_argument("--llm_output", required=True, help="debug_* .llm_output.context.json")
    ap.add_argument("--bundle_json", default=None, help="bundle_pages-*.json (optional)")
    ap.add_argument("--gold_csv", default=None, help="CSV of gold boundaries, one int per line, include 0 and M (optional)")
    ap.add_argument("--desired_chunks", type=int, default=None, help="Requested chunk count (optional; for window sizing)")
    ap.add_argument("--out_dir", default=None, help="Output folder; default debu_files/eval_YYYY_MM_DD_HH_MM")
    args = ap.parse_args()

    evaluate(args.paragraphs, args.llm_output, args.bundle_json, args.gold_csv, args.desired_chunks, args.out_dir)
