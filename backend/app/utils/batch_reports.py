# app/utils/batch_reports.py
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_BASE = Path("backend/testdata")
FLOAT_FIELDS = {
    "precision",
    "recall",
    "f1",
    "mabo",
    "chunk_diff_pct",
    "chunk_size_diff_avg",
    "chunk_size_diff_pct",
    "chunk_time_sec",
    "topic_entropy",
    "offset_p50",
    "offset_p90",
}
INT_FIELDS = {
    "run_index",
    "subpdf_count",
    "pages_total",
    "items_total",
    "desired_chunks",
    "produced_chunks",
    "paragraphs",
    "chunk_diff",
    "too_small",
    "too_big",
    "tp",
    "fp",
    "fn",
    "topic_diversity",
    "duplicate_count",
}


def _as_number(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _find_first(testrun_dir: Path, prefix: str, exts: List[str]) -> Optional[Path]:
    for ext in exts:
        p = testrun_dir / f"{prefix}{ext}"
        if p.exists():
            return p
    for ext in exts:
        candidates = sorted(testrun_dir.glob(f"{prefix}_*{ext}"))
        if candidates:
            return candidates[0]
    return None


def _load_summary_rows(testrun_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary_path = _find_first(testrun_dir, "summary_all", [".csv"])
    meta_path = _find_first(testrun_dir, "testrun_meta", [".json"])
    meta: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if not summary_path or not summary_path.exists():
        return rows, meta

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Dict[str, Any] = dict(raw)
            for key in FLOAT_FIELDS:
                if key in row:
                    num = _as_number(row[key])
                    if num is not None:
                        row[key] = float(num)
            for key in INT_FIELDS:
                if key in row:
                    num = _as_number(row[key])
                    if num is not None:
                        row[key] = int(round(num))
            rows.append(row)
    return rows, meta


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(arr)), float(np.std(arr)))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_box(groups: Dict[str, List[Dict[str, Any]]], metric: str, out_path: Path) -> None:
    data = []
    labels = []
    for g, rows in groups.items():
        vals = [r.get(metric) for r in rows if r.get(metric) is not None]
        if not vals:
            continue
        data.append(vals)
        labels.append(g)
    if not data:
        return
    plt.figure(figsize=(max(6, len(labels) * 1.4), 4.5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} distribution by batch subgroup")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pages_vs_metric(groups: Dict[str, List[Dict[str, Any]]], metric: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for label, rows in groups.items():
        pairs = [
            (r.get("pages_total"), r.get(metric))
            for r in rows
            if r.get("pages_total") is not None and r.get(metric) is not None
        ]
        if not pairs:
            continue
        x, y = zip(*pairs)
        plt.scatter(x, y, alpha=0.6, label=label)
        mean_x, _ = _mean_std(x)
        mean_y, std_y = _mean_std(y)
        plt.errorbar([mean_x], [mean_y], yerr=[std_y], fmt="o", color="black", capsize=4, alpha=0.7)
    if not plt.gca().has_data():
        plt.close()
        return
    plt.xlabel("Pages per run (total)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs pages (mean ± SD)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_overall_bar(rows: List[Dict[str, Any]], out_path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    means = []
    stds = []
    for m in metrics:
        vals = [r.get(m) for r in rows if r.get(m) is not None]
        mu, sd = _mean_std(vals)
        means.append(mu)
        stds.append(sd)
    if all(v == 0 for v in means):
        return
    x = np.arange(len(metrics))
    plt.figure(figsize=(5.5, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylabel("Score")
    plt.title("Overall PRF1 (mean ± SD) across batch")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def aggregate_batch(
    batch_label: str,
    base_dir: Path | str = DEFAULT_BASE,
    testrun_dirs: Optional[List[str | Path]] = None,
) -> Dict[str, Any]:
    """
    Combine multiple testruns (summary_all.csv) under one batch label and emit plots.
    If testrun_dirs is None, we auto-discover runs in base_dir whose testrun_meta.json
    has batch_label set.
    """
    label = (batch_label or "").strip()
    if not label:
        raise ValueError("batch_label is required")

    base_path = Path(base_dir)
    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    if testrun_dirs:
        for td in testrun_dirs:
            p = Path(td)
            meta = {}
            if p.is_dir():
                _, meta = _load_summary_rows(p)
            candidates.append((p, meta))
    else:
        for td in sorted(base_path.glob("testrun_*")):
            if not td.is_dir():
                continue
            _, meta = _load_summary_rows(td)
            if (meta.get("batch_label") or "") == label:
                candidates.append((td, meta))

    seen = set()
    filtered: List[Tuple[Path, Dict[str, Any]]] = []
    for td, meta in candidates:
        if not td.exists():
            continue
        key = td.resolve()
        if key in seen:
            continue
        seen.add(key)
        filtered.append((td, meta))

    all_rows: List[Dict[str, Any]] = []
    used_dirs: List[str] = []
    for td, meta in filtered:
        rows, meta_loaded = _load_summary_rows(td)
        meta = meta or meta_loaded
        if not rows:
            continue
        used_dirs.append(str(td.resolve()))
        label_val = meta.get("batch_label") or label
        variant_val = meta.get("batch_variant")
        for r in rows:
            r["testrun_dir"] = str(td.resolve())
            r["testrun_name"] = td.name
            r["batch_label"] = label_val
            r["batch_variant"] = variant_val
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No summary rows found for batch '{label}'")

    out_dir = base_path / "batch_reports" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "batch_summary_all.csv", all_rows)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        key = r.get("batch_variant") or r.get("batch_label") or "unlabeled"
        groups[key].append(r)

    stats_rows: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        mu_p, sd_p = _mean_std([r.get("precision") for r in rows])
        mu_r, sd_r = _mean_std([r.get("recall") for r in rows])
        mu_f, sd_f = _mean_std([r.get("f1") for r in rows])
        mu_pages, sd_pages = _mean_std([r.get("pages_total") for r in rows])
        stats_rows.append({
            "group": key,
            "runs": len(rows),
            "precision_mean": mu_p,
            "precision_sd": sd_p,
            "recall_mean": mu_r,
            "recall_sd": sd_r,
            "f1_mean": mu_f,
            "f1_sd": sd_f,
            "pages_total_mean": mu_pages,
            "pages_total_sd": sd_pages,
        })
    _write_csv(out_dir / "group_stats.csv", stats_rows)

    # plots
    _plot_metric_box(groups, "f1", out_dir / "box_f1_by_group.png")
    _plot_metric_box(groups, "precision", out_dir / "box_precision_by_group.png")
    _plot_metric_box(groups, "recall", out_dir / "box_recall_by_group.png")
    _plot_pages_vs_metric(groups, "f1", out_dir / "pages_vs_f1.png")
    _plot_pages_vs_metric(groups, "precision", out_dir / "pages_vs_precision.png")
    _plot_pages_vs_metric(groups, "recall", out_dir / "pages_vs_recall.png")
    _plot_overall_bar(all_rows, out_dir / "overall_prf1.png")

    (out_dir / "README.txt").write_text(
        "\n".join([
            f"Batch label: {label}",
            f"Testruns combined: {len(used_dirs)}",
            f"Rows aggregated: {len(all_rows)}",
            f"Groups: {', '.join(sorted(groups.keys()))}",
        ]),
        encoding="utf-8",
    )

    return {
        "batch_label": label,
        "testruns_used": used_dirs,
        "rows_combined": len(all_rows),
        "groups": sorted(groups.keys()),
        "out_dir": str(out_dir.resolve()),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate multiple testruns into a batch report.")
    parser.add_argument("--label", required=True, help="Batch label to aggregate.")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Base testdata directory.")
    parser.add_argument("--testrun", action="append", help="Explicit testrun directory to include (skip auto-discovery).")
    args = parser.parse_args()
    result = aggregate_batch(args.label, base_dir=Path(args.base), testrun_dirs=args.testrun)
    print(json.dumps(result, indent=2))
