import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR_C = DATA_DIR / "C_topic_structure_analysis"
OUT_DIR_C.mkdir(exist_ok=True)
CSV_PATTERNS = ["summary_all*.csv", "summary*.csv", "*merged*.csv"]
META_PREFIX = "testrun_meta_"
MIN_VALID_CHUNKS = 4                                        
COL_PAGES = "pages_total"
COL_ITEMS = "items_total"
COL_PREC  = "precision"
COL_REC   = "recall"
COL_F1    = "f1"
COL_MABO  = "mabo"
COL_TIME  = "chunk_time_sec"
COL_PRODUCED = "produced_chunks"
COL_TOO_SMALL = "too_small"
COL_TOO_BIG = "too_big"
OUTLIER_COLS = [
    COL_F1, COL_REC, COL_MABO, COL_TIME,
    COL_PRODUCED, "size_violation_rate",
]

def require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def relativize(path: Path, bases: list[Path]) -> str:
    """Return a nice relative path w.r.t. the first matching base, else absolute."""
    for base in bases:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)

def drop_degenerate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove failed/degenerate runs where produced_chunks <= 3."""
    if COL_PRODUCED not in df.columns:
        return df
    before = len(df)
    produced = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0)
    df = df[produced >= MIN_VALID_CHUNKS].copy()
    after = len(df)
    print(f"[CLEAN] Dropped {before - after} degenerate runs (produced_chunks <= 3)")
    return df

def drop_iqr_outliers(df: pd.DataFrame, cols: list[str], group_col: str | None = None) -> pd.DataFrame:
    """
    Drop outliers using the IQR rule per column.
    If group_col is provided, compute IQR bounds within each group (recommended).
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    def _filter_block(block: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=block.index)
        for c in cols:
            s = pd.to_numeric(block[c], errors="coerce")
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            mask &= (s.isna() | ((s >= lo) & (s <= hi)))
        return block[mask].copy()
    before = len(df)
    if group_col and group_col in df.columns:
        blocks = []
        for g, block in df.groupby(group_col):
            filtered = _filter_block(block)
            filtered[group_col] = g
            blocks.append(filtered)
        df = pd.concat(blocks, ignore_index=True) if blocks else df.iloc[0:0].copy()
    else:
        df = _filter_block(df)
    after = len(df)
    print(f"[CLEAN] Dropped {before - after} IQR outliers (cols={cols}, group={group_col})")
    return df

def save_boxplot(df: pd.DataFrame, group_col: str, metric: str, ylabel: str, out_path: Path, title: str | None = None):
    groups = sorted(df[group_col].dropna().unique())
    series = []
    labels = []
    for g in groups:
        vals = pd.to_numeric(df.loc[df[group_col] == g, metric], errors="coerce").dropna().values
        if len(vals) > 0:
            series.append(vals)
            labels.append(str(int(g)) if float(g).is_integer() else str(g))
    if not series:
        print(f"[WARN] No data for {metric} boxplot.")
        return
    plt.figure(figsize=(7.5, 4))
    plt.boxplot(series, tick_labels=labels, showfliers=True)
    plt.xlabel("Number of topics" if group_col == "topic_count" else group_col)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_scatter_with_jitter(df: pd.DataFrame, x_fixed: str, y: str, out_path: Path, xlabel: str, ylabel: str, title: str | None = None):
    """
    Scatter plot where x is a small set of discrete groups; add jitter so points are visible.
    """
    if df.empty:
        return
    x = pd.to_numeric(df[x_fixed], errors="coerce")
    yv = pd.to_numeric(df[y], errors="coerce")
    jitter = np.random.normal(loc=0.0, scale=0.06, size=len(df))
    plt.figure(figsize=(7.5, 4))
    plt.scatter(x + jitter, yv, s=18, alpha=0.65)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_bar_by_topic(df: pd.DataFrame, metric: str, ylabel: str, filename: str | Path, title: str | None = None):
    """Bar chart of mean metric per topic_count (expected 1 vs 8)."""
    if "topic_count" not in df.columns:
        print(f"[WARN] topic_count not present, skipping {metric} bar plot.")
        return
    if metric not in df.columns:
        print(f"[WARN] {metric} not present, skipping bar plot.")
        return
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    agg = tmp.groupby("topic_count")[metric].mean(numeric_only=True).sort_index()
    if agg.empty:
        print(f"[WARN] No data for {metric} bar plot.")
        return
    x = np.arange(len(agg.index))
    plt.figure(figsize=(7, 4))
    plt.bar(x, agg.values, width=0.5)
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Topic count")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_desired_vs_produced_by_topic(df: pd.DataFrame, filename: str | Path):
    """Grouped bar: desired (items_total) vs produced_chunks per topic_count."""
    if "topic_count" not in df.columns:
        print("[WARN] topic_count not present, skipping desired vs produced plot.")
        return
    if COL_ITEMS not in df.columns or COL_PRODUCED not in df.columns:
        print("[WARN] Missing items_total or produced_chunks, skipping desired vs produced plot.")
        return
    tmp = df.copy()
    tmp[COL_ITEMS] = pd.to_numeric(tmp[COL_ITEMS], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")
    agg = (
        tmp.groupby("topic_count")[[COL_ITEMS, COL_PRODUCED]]
        .mean(numeric_only=True)
        .sort_index()
    )
    if agg.empty:
        print("[WARN] No grouped data for desired vs produced plot.")
        return
    x = np.arange(len(agg.index))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, agg[COL_ITEMS].values, width, label="Desired (items_total)")
    plt.bar(x + width/2, agg[COL_PRODUCED].values, width, label="Produced (chunks)")
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Topic count")
    plt.ylabel("Mean chunks per run")
    plt.title("Desired vs produced chunk counts by topic count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_items_vs_chunks_by_topic(df: pd.DataFrame, filename: str | Path):
    """
    Scatter of items_total vs produced_chunks.
    - Plots all per-run points for topic_count=1 (orange) and topic_count=8 (blue).
    - Overlays the per-group mean point in black with labels (in front).
    - Adds a legend for the per-run point colors.
    """
    if "topic_count" not in df.columns:
        print("[WARN] topic_count not present, skipping items vs chunks plot.")
        return
    if COL_ITEMS not in df.columns or COL_PRODUCED not in df.columns:
        print("[WARN] Missing items_total or produced_chunks, skipping items vs chunks plot.")
        return
    tmp = df.copy()
    tmp[COL_ITEMS] = pd.to_numeric(tmp[COL_ITEMS], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")
    tmp["topic_count"] = pd.to_numeric(tmp["topic_count"], errors="coerce")
    plot_df = tmp[tmp["topic_count"].isin([1, 8])].dropna(subset=[COL_ITEMS, COL_PRODUCED, "topic_count"]).copy()
    if plot_df.empty:
        print("[WARN] No 1-topic/8-topic rows found for items vs chunks plot.")
        return
    plt.figure(figsize=(6, 5))
    g1 = plot_df[plot_df["topic_count"] == 1]
    g8 = plot_df[plot_df["topic_count"] == 8]
    h1 = None
    h8 = None
    if not g1.empty:
        h1 = plt.scatter(
            g1[COL_ITEMS],
            g1[COL_PRODUCED],
            s=45,
            alpha=0.6,
            color="orange",
            edgecolors="none",
            label="1 topic",
            zorder=2,
        )
    if not g8.empty:
        h8 = plt.scatter(
            g8[COL_ITEMS],
            g8[COL_PRODUCED],
            s=45,
            alpha=0.6,
            color="#1f77b4",
            edgecolors="none",
            label="8 topics",
            zorder=2,
        )
    agg = (
        plot_df.groupby("topic_count")[[COL_ITEMS, COL_PRODUCED]]
        .mean(numeric_only=True)
        .sort_index()
    )
    plt.scatter(
        agg[COL_ITEMS],
        agg[COL_PRODUCED],
        s=110,
        color="black",
        zorder=5,
    )
    for t, row in agg.iterrows():
        plt.annotate(
            f"{int(t)} topics",
            (row[COL_ITEMS], row[COL_PRODUCED]),
            textcoords="offset points",
            xytext=(6, 4),
            color="black",
            fontsize=10,
            zorder=6,
        )
    lo = min(agg[COL_ITEMS].min(), agg[COL_PRODUCED].min())
    hi = max(agg[COL_ITEMS].max(), agg[COL_PRODUCED].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean ground-truth items (items_total)")
    plt.ylabel("Mean produced chunks")
    plt.title("Produced chunks vs ground-truth items (per-run + mean by topic count)")
    handles = [h for h in [h1, h8] if h is not None]
    labels = [h.get_label() for h in handles]
    if handles:
        plt.legend(handles, labels, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_items_vs_chunks_1_vs_8_topics(df: pd.DataFrame, filename: str | Path):
    """
    Scatter comparing items_total vs produced_chunks for 1 vs 8 topics (topic-structure experiment).
    Uses mean per topic_count; if one of the groups is missing, skip with a warning.
    """
    required = {1, 8}
    if COL_ITEMS not in df.columns or COL_PRODUCED not in df.columns or "topic_count" not in df.columns:
        print("[WARN] Missing items_total/produced_chunks/topic_count; skipping 1 vs 8 scatter.")
        return
    tmp = df.copy()
    tmp[COL_ITEMS] = pd.to_numeric(tmp[COL_ITEMS], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")
    agg = (
        tmp[tmp["topic_count"].isin(required)]
        .groupby("topic_count")[[COL_ITEMS, COL_PRODUCED]]
        .mean(numeric_only=True)
    )
    if set(agg.index) != required:
        print("[WARN] Need both 1-topic and 8-topic data for the scatter; skipping.")
        return
    plt.figure(figsize=(5, 4))
    plt.scatter(agg[COL_ITEMS], agg[COL_PRODUCED], s=80)
    for t, row in agg.iterrows():
        plt.annotate(f"{int(t)} topics", (row[COL_ITEMS], row[COL_PRODUCED]), textcoords="offset points", xytext=(6, 4))
    lo = min(agg[COL_ITEMS].min(), agg[COL_PRODUCED].min())
    hi = max(agg[COL_ITEMS].max(), agg[COL_PRODUCED].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean items_total")
    plt.ylabel("Mean produced_chunks")
    plt.title("Items vs produced chunks (1 vs 8 topics)")
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_chunk_size_diff_avg_by_topic(df: pd.DataFrame, filename: str | Path):
    if "chunk_size_diff_avg" not in df.columns:
        print("[WARN] Missing chunk_size_diff_avg. Skipping plot.")
        return
    tmp = df.copy()
    tmp["chunk_size_diff_avg"] = pd.to_numeric(tmp["chunk_size_diff_avg"], errors="coerce")
    agg = (
        tmp.groupby("topic_count")["chunk_size_diff_avg"]
        .mean()
        .sort_index()
    )
    if agg.empty:
        print("[WARN] No grouped data for chunk_size_diff_avg. Skipping plot.")
        return
    x = np.arange(len(agg.index))
    plt.figure(figsize=(7, 4))
    plt.bar(x, agg.values)
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Topic count")
    plt.ylabel("Mean chunk size deviation (paragraphs)")
    plt.title("Average chunk size deviation by topic count")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_chunk_size_diff_pct_by_topic(df: pd.DataFrame, filename: str | Path):
    if "chunk_size_diff_pct" not in df.columns:
        print("[WARN] Missing chunk_size_diff_pct. Skipping plot.")
        return
    tmp = df.copy()
    tmp["chunk_size_diff_pct"] = pd.to_numeric(tmp["chunk_size_diff_pct"], errors="coerce")
    agg = (
        tmp.groupby("topic_count")["chunk_size_diff_pct"]
        .mean()
        .sort_index()
    ) * 100.0                      
    if agg.empty:
        print("[WARN] No grouped data for chunk_size_diff_pct. Skipping plot.")
        return
    x = np.arange(len(agg.index))
    plt.figure(figsize=(7, 4))
    plt.bar(x, agg.values)
    plt.xticks(x, [str(int(t)) for t in agg.index])
    plt.xlabel("Topic count")
    plt.ylabel("Mean chunk count deviation (%)")
    plt.title("Average chunk size deviation by topic count (%)")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR_C / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Running in:", DATA_DIR)
csv_files = []
for pat in CSV_PATTERNS:
    csv_files.extend(DATA_DIR.rglob(pat))
csv_files = sorted(set(csv_files))
c_candidates = []
for p in csv_files:
    name = p.name.lower()
    if "topics" in name and name.endswith(".csv"):
        c_candidates.append(p)
print(f"Found {len(c_candidates)} topic-structure CSV candidates (using only CSVs next to this script).")
dfs_c = []
needed_c = [COL_F1, COL_REC, COL_MABO, COL_TIME, COL_PRODUCED]
topic_re = re.compile(r"(\d+)\s*topics", re.IGNORECASE)
for csv_path in c_candidates:
    df = pd.read_csv(csv_path)
    require_columns(df, needed_c, context=str(csv_path))
    m = topic_re.search(csv_path.stem.replace("_", " "))
    if not m:
        m = topic_re.search(csv_path.name)
    if not m:
        print(f"[SKIP] Could not infer topic_count from filename: {csv_path.name}")
        continue
    topic_count = int(m.group(1))
    df["topic_count"] = topic_count
    df["source_file"] = relativize(csv_path, [DATA_DIR])
    if COL_TOO_SMALL in df.columns and COL_TOO_BIG in df.columns:
        produced_safe = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0).replace(0, 1)
        df["size_violation_rate"] = (
            pd.to_numeric(df[COL_TOO_SMALL], errors="coerce").fillna(0)
            + pd.to_numeric(df[COL_TOO_BIG], errors="coerce").fillna(0)
        ) / produced_safe
    dfs_c.append(df)
if not dfs_c:
    raise FileNotFoundError(
        "No usable topic-structure CSVs found. Ensure filenames contain e.g. '1Topics', '8Topics', etc."
    )
data_c = pd.concat(dfs_c, ignore_index=True)
print("[C] Rows before cleaning:", len(data_c))
data_c = drop_degenerate_runs(data_c)
data_c = drop_iqr_outliers(data_c, OUTLIER_COLS, group_col="topic_count")
data_c.to_csv(OUT_DIR_C / "merged_C_all_runs.csv", index=False)
print("[C] Rows after cleaning:", len(data_c))
print("[C] Runs per topic_count:\n", data_c.groupby("topic_count").size().sort_index())
summary_c = (
    data_c.groupby("topic_count")
          .agg(
              runs=(COL_F1, "count"),
              pages_mean=(COL_PAGES, "mean") if COL_PAGES in data_c.columns else (COL_F1, "count"),
              f1_mean=(COL_F1, "mean"),
              f1_std=(COL_F1, "std"),
              recall_mean=(COL_REC, "mean"),
              recall_std=(COL_REC, "std"),
              mabo_median=(COL_MABO, "median"),
              mabo_mean=(COL_MABO, "mean"),
              mabo_std=(COL_MABO, "std"),
              produced_mean=(COL_PRODUCED, "mean"),
              produced_std=(COL_PRODUCED, "std"),
              runtime_mean=(COL_TIME, "mean"),
              runtime_std=(COL_TIME, "std"),
          )
          .reset_index()
          .sort_values("topic_count")
)
summary_c.to_csv(OUT_DIR_C / "summary_C_by_topic_count.csv", index=False)
latex_c = summary_c.to_latex(
    index=False,
    float_format=lambda x: f"{x:.4f}",
    caption="Topic-structure experiment summary grouped by number of topics (mean$\\pm$std; MABO reported as median).",
    label="tab:topic_structure_summary",
)
(OUT_DIR_C / "table_C_topic_structure_summary.tex").write_text(latex_c, encoding="utf-8")
save_boxplot(
    data_c, "topic_count", COL_F1, "F1 score",
    OUT_DIR_C / "C_f1_box_by_topic_count.png",
    title="F1 score by topic diversity (after cleaning)"
)
save_boxplot(
    data_c, "topic_count", COL_MABO, "Mean absolute boundary offset (paragraphs)",
    OUT_DIR_C / "C_mabo_box_by_topic_count.png",
    title="Boundary offset by topic diversity (after cleaning)"
)
save_boxplot(
    data_c, "topic_count", COL_PRODUCED, "Produced chunks",
    OUT_DIR_C / "C_produced_chunks_box_by_topic_count.png",
    title="Produced chunk count by topic diversity (after cleaning)"
)
save_boxplot(
    data_c, "topic_count", COL_REC, "Recall",
    OUT_DIR_C / "C_recall_box_by_topic_count.png",
    title="Recall by topic diversity (after cleaning)"
)
save_scatter_with_jitter(
    data_c, "topic_count", COL_F1,
    OUT_DIR_C / "C_f1_scatter_jitter.png",
    xlabel="Number of topics", ylabel="F1 score"
)
plot_bar_by_topic(
    data_c, COL_PRODUCED, "Mean produced chunks",
    "C_produced_chunks_bar_by_topic.png",
    title="Produced chunks by topic count"
)
plot_bar_by_topic(
    data_c, COL_F1, "F1 score",
    "C_f1_bar_by_topic.png",
    title="F1 score by topic count"
)
plot_bar_by_topic(
    data_c, COL_REC, "Recall",
    "C_recall_bar_by_topic.png",
    title="Recall by topic count"
)
plot_items_vs_chunks_by_topic(data_c, "C_items_vs_chunks_by_topic.png")
plot_desired_vs_produced_by_topic(data_c, "C_desired_vs_produced_by_topic.png")
plot_chunk_size_diff_avg_by_topic(data_c, "C_chunk_size_diff_avg_by_topic.png")
plot_chunk_size_diff_pct_by_topic(data_c, "C_chunk_size_diff_pct_by_topic.png")
print(f"Done.\n - Outputs: {OUT_DIR_C.resolve()}")
