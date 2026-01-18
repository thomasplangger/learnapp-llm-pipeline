import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "D_structure_analysis"
OUT_DIR.mkdir(exist_ok=True)
CSV_PATTERNS = ["summary_all*.csv", "summary*.csv", "*merged*.csv"]
MIN_VALID_CHUNKS = 4                                        
COL_ITEMS = "items_total"
COL_PRODUCED = "produced_chunks"
COL_TOO_SMALL = "too_small"
COL_TOO_BIG = "too_big"
COL_RECALL = "recall"
COL_F1 = "f1"
COL_MABO = "mabo"
COL_CHUNK_DIFF_AVG = "chunk_size_diff_avg"
COL_CHUNK_DIFF_PCT = "chunk_size_diff_pct"
CONDITION_ORDER = ["short", "medium", "long"]

def require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def normalize_condition(val: str) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    if "short" in s:
        return "short"
    if "medium" in s:
        return "medium"
    if "long" in s:
        return "long"
    return s

def condition_from_filename(path: Path) -> str | None:
    name = path.stem.lower()
    for key in ["short", "medium", "long"]:
        if key in name:
            return key
    return None

def ordered_conditions(values: list[str]) -> list[str]:
    extras = sorted([v for v in values if v not in CONDITION_ORDER])
    return [v for v in CONDITION_ORDER if v in values] + extras

def drop_degenerate_runs(df: pd.DataFrame) -> pd.DataFrame:
    if COL_PRODUCED not in df.columns:
        return df
    before = len(df)
    produced = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0)
    df = df[produced >= MIN_VALID_CHUNKS].copy()
    after = len(df)
    print(f"[CLEAN] Dropped {before - after} degenerate runs (produced_chunks <= 3)")
    return df

def ensure_size_violation_rate(df: pd.DataFrame) -> pd.DataFrame:
    if COL_TOO_SMALL not in df.columns or COL_TOO_BIG not in df.columns:
        return df
    produced_safe = pd.to_numeric(df[COL_PRODUCED], errors="coerce").fillna(0).replace(0, 1)
    df["size_violation_rate"] = (
        pd.to_numeric(df[COL_TOO_SMALL], errors="coerce").fillna(0)
        + pd.to_numeric(df[COL_TOO_BIG], errors="coerce").fillna(0)
    ) / produced_safe
    return df

def save_boxplot(df: pd.DataFrame, metric: str, ylabel: str, filename: str, title: str):
    conditions = ordered_conditions(sorted(df["condition"].dropna().unique().tolist()))
    series = []
    labels = []
    for cond in conditions:
        vals = pd.to_numeric(df.loc[df["condition"] == cond, metric], errors="coerce").dropna().values
        if len(vals) > 0:
            series.append(vals)
            labels.append(cond)
    if not series:
        print(f"[WARN] No data for {metric} boxplot.")
        return
    plt.figure(figsize=(7.5, 4))
    plt.boxplot(series, tick_labels=labels, showfliers=True)
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_grouped_bars(df: pd.DataFrame, filename: str, title: str):
    if COL_ITEMS not in df.columns or COL_PRODUCED not in df.columns:
        print("[WARN] Missing items_total or produced_chunks; skipping desired vs produced.")
        return
    tmp = df.copy()
    tmp[COL_ITEMS] = pd.to_numeric(tmp[COL_ITEMS], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")
    agg = tmp.groupby("condition")[[COL_ITEMS, COL_PRODUCED]].mean(numeric_only=True)
    if agg.empty:
        print("[WARN] No grouped data for desired vs produced.")
        return
    conditions = ordered_conditions(sorted(agg.index.tolist()))
    agg = agg.loc[conditions]
    x = np.arange(len(conditions))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, agg[COL_ITEMS].values, width, label="Desired (items_total)")
    plt.bar(x + width / 2, agg[COL_PRODUCED].values, width, label="Produced (chunks)")
    plt.xticks(x, conditions)
    plt.xlabel("Condition")
    plt.ylabel("Mean chunks per run")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_items_vs_chunks_scatter(df: pd.DataFrame, filename: str, title: str):
    if COL_ITEMS not in df.columns or COL_PRODUCED not in df.columns:
        print("[WARN] Missing items_total or produced_chunks; skipping items vs chunks.")
        return
    tmp = df.copy()
    tmp[COL_ITEMS] = pd.to_numeric(tmp[COL_ITEMS], errors="coerce")
    tmp[COL_PRODUCED] = pd.to_numeric(tmp[COL_PRODUCED], errors="coerce")
    agg = tmp.groupby("condition")[[COL_ITEMS, COL_PRODUCED]].mean(numeric_only=True)
    if agg.empty:
        print("[WARN] No grouped data for items vs chunks.")
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(agg[COL_ITEMS], agg[COL_PRODUCED], s=90)
    for cond, row in agg.iterrows():
        plt.annotate(cond, (row[COL_ITEMS], row[COL_PRODUCED]), textcoords="offset points", xytext=(6, 4))
    lo = min(agg[COL_ITEMS].min(), agg[COL_PRODUCED].min())
    hi = max(agg[COL_ITEMS].max(), agg[COL_PRODUCED].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean ground-truth items (items_total)")
    plt.ylabel("Mean produced chunks")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_bar_by_condition(df: pd.DataFrame, metric: str, ylabel: str, filename: str, title: str, scale_pct: bool = False):
    if metric not in df.columns:
        print(f"[WARN] Missing {metric}; skipping.")
        return
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    agg = tmp.groupby("condition")[metric].mean(numeric_only=True)
    if agg.empty:
        print(f"[WARN] No grouped data for {metric}.")
        return
    if scale_pct:
        agg = agg * 100.0
    conditions = ordered_conditions(sorted(agg.index.tolist()))
    agg = agg.loc[conditions]
    x = np.arange(len(conditions))
    plt.figure(figsize=(7, 4))
    plt.bar(x, agg.values)
    plt.xticks(x, conditions)
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Running in:", DATA_DIR)
csv_files = []
for pat in CSV_PATTERNS:
    csv_files.extend(DATA_DIR.rglob(pat))
csv_files = sorted(set(csv_files))
if not csv_files:
    raise FileNotFoundError("No CSV files found next to this script.")
dfs = []
required_any = [COL_PRODUCED, COL_RECALL, COL_F1, COL_MABO]
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    require_columns(df, required_any, context=str(csv_path))
    condition = None
    if "condition" in df.columns:
        condition = df["condition"].dropna().astype(str).map(normalize_condition)
        if condition.notna().any():
            df["condition"] = condition
        else:
            df["condition"] = None
    if "condition" not in df.columns or df["condition"].isna().all():
        inferred = condition_from_filename(csv_path)
        df["condition"] = inferred
    df["source_file"] = str(csv_path.name)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data["condition"] = data["condition"].map(normalize_condition)
data = data.dropna(subset=["condition"]).copy()
if data.empty:
    raise ValueError("No rows with a valid condition (short/medium/long). Check filenames or add a condition column.")
print("[D] Rows before cleaning:", len(data))
data = drop_degenerate_runs(data)
data = ensure_size_violation_rate(data)
print("[D] Rows after cleaning:", len(data))
print("[D] Runs per condition:\n", data.groupby("condition").size())
data.to_csv(OUT_DIR / "merged_D_all_runs.csv", index=False)
save_boxplot(
    data, COL_PRODUCED, "Produced chunks",
    "D1_produced_chunks_box_by_condition.png",
    title="Produced chunks per condition"
)
save_boxplot(
    data, "size_violation_rate", "Size violation rate",
    "D2_size_violation_rate_box_by_condition.png",
    title="Size violation rate per condition"
)
save_boxplot(
    data, COL_RECALL, "Recall",
    "D3_recall_box_by_condition.png",
    title="Recall per condition"
)
save_boxplot(
    data, COL_F1, "F1 score",
    "D4_f1_box_by_condition.png",
    title="F1 score per condition"
)
save_boxplot(
    data, COL_MABO, "Mean absolute boundary offset (paragraphs)",
    "D5_mabo_box_by_condition.png",
    title="Boundary offset per condition"
)
save_grouped_bars(
    data,
    "D_desired_vs_produced_chunks.png",
    title="Desired vs produced chunks by condition"
)
save_items_vs_chunks_scatter(
    data,
    "D_items_vs_chunks_by_condition.png",
    title="Produced chunks vs ground-truth items (by condition)"
)
save_bar_by_condition(
    data, COL_CHUNK_DIFF_AVG, "Mean chunk size deviation (paragraphs)",
    "D_chunk_size_diff_avg_paras.png",
    title="Average chunk size deviation by condition"
)
save_bar_by_condition(
    data, COL_CHUNK_DIFF_PCT, "Mean chunk count deviation (%)",
    "D_chunk_size_diff_pct_by_condition.png",
    title="Average chunk size deviation by condition (%)",
    scale_pct=True
)
print(f"Done.\n - Outputs: {OUT_DIR.resolve()}")
