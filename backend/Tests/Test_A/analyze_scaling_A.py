import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "A_scaling_analysis"
OUT_DIR.mkdir(exist_ok=True)

CSV_PATTERNS = ["summary_all*.csv", "summary*.csv"]
META_PREFIX = "testrun_meta_"
MIN_VALID_CHUNKS = 4

COL_PAGES = "pages_total"
COL_PREC = "precision"
COL_REC  = "recall"
COL_F1   = "f1"
COL_MABO = "mabo"
COL_TIME = "chunk_time_sec"
COL_PRODUCED = "produced_chunks"
COL_TOO_SMALL = "too_small"
COL_TOO_BIG = "too_big"

COL_ITEMS_TOTAL = "items_total"
COL_IN_WINDOW = "in_window"

TOPIC_COL_PREFIXES = ["topic_", "gt_topic_", "groundtruth_"]

TOPIC_LIST_CANDIDATES = [
    "topics", "topic_list", "topics_list", "topic_names",
    "topics_per_pdf", "pdf_topics", "topic_sequence"
]

def require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def read_meta_for_csv(csv_path: Path) -> dict:
    stem = csv_path.stem
    if "summary_all_" in stem:
        suffix = stem.replace("summary_all_", "")
    elif "summary_" in stem:
        suffix = stem.replace("summary_", "")
    else:
        suffix = stem.split("_")[-1]

    meta_path = csv_path.with_name(f"{META_PREFIX}{suffix}.json")
    print(f"[META] {csv_path.name} -> {meta_path.name} (exists={meta_path.exists()})")

    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))

def extract_target_pages(meta: dict) -> int | None:
    if not meta:
        return None

    for k in ["min_pages", "pages_min", "target_pages"]:
        if k in meta and meta[k] is not None:
            try:
                return int(meta[k])
            except Exception:
                pass

    for parent in ["inputs", "config", "params", "settings", "request", "job", "plan"]:
        obj = meta.get(parent)
        if isinstance(obj, dict):
            for k in ["min_pages", "pages_min", "target_pages"]:
                if k in obj and obj[k] is not None:
                    try:
                        return int(obj[k])
                    except Exception:
                        pass
    return None

def fallback_target_pages_from_filename(csv_path: Path) -> int | None:
    name = csv_path.stem.lower()
    if "a1" in name: return 5
    if "a2" in name: return 10
    if "a3" in name: return 20
    if "a4" in name: return 40
    if "a5" in name: return 80
    return None

def save_boxplot(df: pd.DataFrame, metric: str, ylabel: str, filename: str):
    pairs = []
    for t in sorted(df["target_pages"].dropna().unique()):
        vals = pd.to_numeric(df.loc[df["target_pages"] == t, metric], errors="coerce").dropna().values
        if len(vals) > 0:
            pairs.append((t, vals))

    if not pairs:
        print(f"[WARN] No data for boxplot {metric}, skipping {filename}")
        return

    order = [p[0] for p in pairs]
    series = [p[1] for p in pairs]

    plt.figure()
    plt.boxplot(series, tick_labels=[str(int(x)) for x in order], showfliers=True)
    plt.xlabel("Target pages (generator setting)")
    plt.ylabel(ylabel)
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_mean_std_trend(df: pd.DataFrame, metric: str, ylabel: str, filename: str):
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    g = tmp.groupby("target_pages")[metric]
    mean = g.mean()
    std = g.std()

    if mean.empty:
        print(f"[WARN] No data for trend {metric}, skipping {filename}")
        return

    plt.figure()
    plt.errorbar(mean.index, mean.values, yerr=std.values, fmt="o-")
    plt.xlabel("Target pages (generator setting)")
    plt.ylabel(ylabel)
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_scatter(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, filename: str):
    if df.empty:
        print(f"[WARN] No data for scatter {y} vs {x}, skipping {filename}")
        return

    plt.figure()
    plt.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), s=18)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def drop_outlier_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove failed/degenerate runs where produced_chunks <= 3.
    """
    if COL_PRODUCED not in df.columns:
        return df
    before = len(df)
    produced_clean = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0)
    df = df[produced_clean >= MIN_VALID_CHUNKS]
    after = len(df)
    print(f"[CLEAN] Removed {before - after} outlier runs (produced_chunks <= 3)")
    return df

def detect_numeric_topic_columns(df: pd.DataFrame) -> list[str]:
    """Return topic_* columns that are actually numeric (or numeric-convertible)."""
    candidates = []
    for c in df.columns:
        if any(c.startswith(p) for p in TOPIC_COL_PREFIXES):
            candidates.append(c)

    numeric_topics = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_topics.append(c)

    return numeric_topics

def try_build_topic_count_columns_from_list(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    If there are no numeric topic columns, try to find a column that contains
    a list of topics per run and expand it into numeric count columns.
    """
    topic_col = None
    for cand in TOPIC_LIST_CANDIDATES:
        if cand in df.columns:
            topic_col = cand
            break
    if topic_col is None:
        return df, []

    def parse_topics(val):
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            if "," in val:
                return [x.strip() for x in val.split(",") if x.strip()]
        return []

    topics_parsed = df[topic_col].apply(parse_topics)
    all_topics = sorted({t for lst in topics_parsed for t in lst})

    if not all_topics:
        return df, []                         
    for t in all_topics:
        colname = f"topic_{t}"
        df[colname] = topics_parsed.apply(lambda lst: sum(1 for x in lst if x == t))

    return df, [f"topic_{t}" for t in all_topics]

def plot_desired_vs_produced_chunks(df: pd.DataFrame, filename: str):
    if "items_total" not in df.columns or "produced_chunks" not in df.columns:
        print("[WARN] Missing items_total or produced_chunks. Skipping desired vs produced plot.")
        return

    tmp = df.copy()
    tmp["items_total"] = pd.to_numeric(tmp["items_total"], errors="coerce")
    tmp["produced_chunks"] = pd.to_numeric(tmp["produced_chunks"], errors="coerce")

    agg = (
        tmp.groupby("target_pages")[["items_total", "produced_chunks"]]
        .mean()
        .sort_index()
    )

    x = np.arange(len(agg.index))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, agg["items_total"], width, label="Desired (reference)")
    plt.bar(x + width/2, agg["produced_chunks"], width, label="Produced")

    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Target pages")
    plt.ylabel("Mean chunks per run")
    plt.title("Desired vs produced chunk counts across scaling conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_chunk_size_diff_avg_by_target(df: pd.DataFrame, filename: str):
    if "chunk_size_diff_avg" not in df.columns:
        print("[WARN] Missing chunk_size_diff_avg. Skipping plot.")
        return

    tmp = df.copy()
    tmp["chunk_size_diff_avg"] = pd.to_numeric(tmp["chunk_size_diff_avg"], errors="coerce")

    agg = (
        tmp.groupby("target_pages")["chunk_size_diff_avg"]
        .mean()
        .sort_index()
    )

    x = np.arange(len(agg.index))

    plt.figure(figsize=(10, 4))
    plt.bar(x, agg.values)
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Target pages")
    plt.ylabel("Mean chunk size deviation (paragraphs)")
    plt.title("Average chunk size deviation from desired size (paragraphs)")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_chunk_size_diff_pct_by_target(df: pd.DataFrame, filename: str):
    if "chunk_size_diff_pct" not in df.columns:
        print("[WARN] Missing chunk_size_diff_pct. Skipping plot.")
        return

    tmp = df.copy()
    tmp["chunk_size_diff_pct"] = pd.to_numeric(tmp["chunk_size_diff_pct"], errors="coerce")

    agg = (
        tmp.groupby("target_pages")["chunk_size_diff_pct"]
        .mean()
        .sort_index()
    ) * 100.0

    x = np.arange(len(agg.index))

    plt.figure(figsize=(10, 4))
    plt.bar(x, agg.values)
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Target pages")
    plt.ylabel("Mean chunk count deviation (%)")
    plt.title("Average chunk size deviation from desired (%)")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Running in:", DATA_DIR)

csv_files = []
for pat in CSV_PATTERNS:
    csv_files.extend(DATA_DIR.rglob(pat))

csv_files = [p for p in csv_files if OUT_DIR not in p.parents]
csv_files = sorted(set(csv_files))

print(f"Found {len(csv_files)} CSV files:")
for p in csv_files[:30]:
    print(" -", p.relative_to(DATA_DIR))
if not csv_files:
    raise FileNotFoundError("No summary CSVs found in this folder or subfolders.")


dfs = []
needed_cols = [
    COL_PAGES, COL_PREC, COL_REC, COL_F1, COL_MABO,
    COL_TIME, COL_PRODUCED, COL_TOO_SMALL, COL_TOO_BIG
]

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    require_columns(df, needed_cols, context=str(csv_path))

    meta = read_meta_for_csv(csv_path)
    target_pages = extract_target_pages(meta)
    if target_pages is None:
        target_pages = fallback_target_pages_from_filename(csv_path)

    df["source_file"] = str(csv_path.relative_to(DATA_DIR))
    df["target_pages"] = float(target_pages) if target_pages is not None else None

    produced_safe = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0).replace(0, 1)
    df["size_violation_rate"] = (pd.to_numeric(df[COL_TOO_SMALL], errors="coerce").fillna(0)
                                 + pd.to_numeric(df[COL_TOO_BIG], errors="coerce").fillna(0)) / produced_safe

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = drop_outlier_runs(data)
data.to_csv(OUT_DIR / "merged_all_runs.csv", index=False)

print("Rows total:", len(data))
print("Rows with target_pages:", data["target_pages"].notna().sum())
print("Unique target_pages:", sorted(data["target_pages"].dropna().unique()))

grouped = data.dropna(subset=["target_pages"]).copy()
summary = (
    grouped.groupby("target_pages")
           .agg(
               runs=(COL_F1, "count"),
               pages_mean=(COL_PAGES, "mean"),
               pages_min=(COL_PAGES, "min"),
               pages_max=(COL_PAGES, "max"),
               precision_mean=(COL_PREC, "mean"),
               precision_std=(COL_PREC, "std"),
               recall_mean=(COL_REC, "mean"),
               recall_std=(COL_REC, "std"),
               f1_mean=(COL_F1, "mean"),
               f1_std=(COL_F1, "std"),
               mabo_median=(COL_MABO, "median"),
               runtime_mean=(COL_TIME, "mean"),
               runtime_std=(COL_TIME, "std"),
               size_viol_mean=("size_violation_rate", "mean"),
               size_viol_std=("size_violation_rate", "std"),
           )
           .reset_index()
           .sort_values("target_pages")
)
summary.to_csv(OUT_DIR / "summary_by_target_pages.csv", index=False)

latex = summary.assign(target_pages=summary["target_pages"].astype(int)).to_latex(
    index=False,
    float_format=lambda x: f"{x:.4f}",
    caption="Scaling experiment summary grouped by target page setting (mean$\\pm$std; MABO reported as median).",
    label="tab:scaling_summary",
)
(OUT_DIR / "table_scaling_summary.tex").write_text(latex, encoding="utf-8")


save_boxplot(grouped, COL_F1, "F1 score", "A_f1_box_by_target.png")
save_boxplot(grouped, COL_PREC, "Precision", "A_precision_box_by_target.png")
save_boxplot(grouped, COL_REC, "Recall", "A_recall_box_by_target.png")
save_boxplot(grouped, COL_MABO, "MABO", "A_mabo_box_by_target.png")
save_boxplot(grouped, COL_TIME, "Runtime (s)", "A_runtime_box_by_target.png")
save_boxplot(grouped, "size_violation_rate", "Size violation rate", "A_sizeviol_box_by_target.png")

save_mean_std_trend(grouped, COL_F1, "F1 score (mean ± std)", "A_f1_trend_by_target.png")
save_mean_std_trend(grouped, COL_REC, "Recall (mean ± std)", "A_recall_trend_by_target.png")
save_mean_std_trend(grouped, COL_TIME, "Runtime (s) (mean ± std)", "A_runtime_trend_by_target.png")

save_scatter(data, COL_PAGES, COL_F1, "Actual pages_total", "F1 score", "A_f1_scatter_by_actual_pages.png")
save_scatter(data, COL_PAGES, COL_REC, "Actual pages_total", "Recall", "A_recall_scatter_by_actual_pages.png")
save_scatter(data, COL_PAGES, COL_PREC, "Actual pages_total", "Precision", "A_precision_scatter_by_actual_pages.png")
save_scatter(data, COL_PAGES, COL_TIME, "Actual pages_total", "Runtime (s)", "A_runtime_scatter_by_actual_pages.png")
save_scatter(data, COL_PAGES, "size_violation_rate", "Actual pages_total", "Size violation rate", "A_sizeviol_scatter_by_actual_pages.png")


def plot_boundaries_by_target(df: pd.DataFrame, filename: str):
    topic_cols = detect_numeric_topic_columns(df)
    if not topic_cols:
        df2, built = try_build_topic_count_columns_from_list(df.copy())
        topic_cols = built
        df = df2

    if not topic_cols:
        print("[WARN] No numeric topic columns (and no parsable topic list). Skipping stacked topic plot.")
        return
    for c in topic_cols + [COL_PRODUCED]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = df.groupby("target_pages")[topic_cols + [COL_PRODUCED]].mean(numeric_only=True).sort_index()

    x = np.arange(len(agg.index))
    plt.figure(figsize=(14, 4))

    bottom = np.zeros(len(agg.index))
    for c in topic_cols:
        label = c
        for p in TOPIC_COL_PREFIXES:
            if label.startswith(p):
                label = label[len(p):]
        plt.bar(x, agg[c].values, bottom=bottom, label=label)
        bottom += agg[c].values

    plt.bar(x + 0.35, agg[COL_PRODUCED].values, width=0.35, label="produced_chunks")

    plt.xticks(x + 0.17, [str(int(t)) for t in agg.index])
    plt.xlabel("Target pages")
    plt.ylabel("Mean count per run")
    plt.title("Reference topic boundaries (stacked) vs produced chunks (mean per target)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_items_vs_chunks_by_target(df: pd.DataFrame, filename: str):
    if COL_ITEMS_TOTAL not in df.columns:
        print(f"[WARN] Missing column '{COL_ITEMS_TOTAL}'. Skipping items vs chunks plot.")
        return

    tmp = df.copy()
    tmp[COL_ITEMS_TOTAL] = pd.to_numeric(tmp[COL_ITEMS_TOTAL], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")

    agg = tmp.groupby("target_pages")[[COL_ITEMS_TOTAL, COL_PRODUCED]].mean(numeric_only=True).sort_index()

    plt.figure()
    plt.scatter(agg[COL_ITEMS_TOTAL], agg[COL_PRODUCED], s=60)

    for t, row in agg.iterrows():
        plt.annotate(str(int(t)), (row[COL_ITEMS_TOTAL], row[COL_PRODUCED]), textcoords="offset points", xytext=(6, 4))

    lo = min(agg[COL_ITEMS_TOTAL].min(), agg[COL_PRODUCED].min())
    hi = max(agg[COL_ITEMS_TOTAL].max(), agg[COL_PRODUCED].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Mean ground-truth topics/items (items_total)")
    plt.ylabel("Mean produced chunks")
    plt.title("Mean produced chunks vs mean ground-truth items (one point per target)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_size_window_by_target(df: pd.DataFrame, filename: str):
    for c in [COL_IN_WINDOW, COL_TOO_SMALL, COL_TOO_BIG]:
        if c not in df.columns:
            print(f"[WARN] Missing column '{c}'. Skipping size-window plot.")
            return

    tmp = df.copy()
    for c in [COL_IN_WINDOW, COL_TOO_SMALL, COL_TOO_BIG]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    agg = tmp.groupby("target_pages")[[COL_IN_WINDOW, COL_TOO_SMALL, COL_TOO_BIG]].mean(numeric_only=True).sort_index()

    x = np.arange(len(agg.index))
    plt.figure(figsize=(10, 4))
    plt.bar(x, agg[COL_IN_WINDOW], label="in-window")
    plt.bar(x, agg[COL_TOO_SMALL], bottom=agg[COL_IN_WINDOW], label="too small")
    plt.bar(x, agg[COL_TOO_BIG], bottom=agg[COL_IN_WINDOW] + agg[COL_TOO_SMALL], label="too big")

    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Target pages")
    plt.ylabel("Mean chunk count per run")
    plt.title("Chunk size window compliance (mean per target)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


plot_desired_vs_produced_chunks(grouped, "A_desired_vs_produced_chunks.png")
plot_items_vs_chunks_by_target(grouped, "A_items_vs_chunks_by_target.png")
plot_size_window_by_target(grouped, "A_size_window_by_target.png")
plot_chunk_size_diff_avg_by_target(grouped, "A_chunk_size_diff_avg_paras.png")
plot_chunk_size_diff_pct_by_target(grouped, "A_chunk_size_diff_pct_by_target.png")



print(f"Done. Outputs written to: {OUT_DIR.resolve()}")
