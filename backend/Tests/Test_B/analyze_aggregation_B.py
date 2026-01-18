import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "B_aggregation_analysis"
OUT_DIR.mkdir(exist_ok=True)
CONFIG_PATTERNS = {
    "B1_1PDFs": "summary_B1_1PDFs",
    "B2_2PDFs": "summary_B2_2PDFs",
    "B3_4PDFs": "summary_B3_4PDFs",
}
COL_F1 = "f1"
COL_REC = "recall"
COL_PREC = "precision"
COL_ITEMS = "items_total"
COL_PRODUCED = "produced_chunks"
COL_RUNTIME = "chunk_time_sec"
COL_CHUNK_DIFF = "chunk_diff"

def _require_cols(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing columns: {missing}. Have: {list(df.columns)}")

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _compute_chunk_diff(df: pd.DataFrame) -> pd.Series:
    if COL_CHUNK_DIFF in df.columns:
        return _safe_num(df[COL_CHUNK_DIFF])
    _require_cols(df, [COL_ITEMS, COL_PRODUCED], "chunk_diff_fallback")
    return (_safe_num(df[COL_ITEMS]) - _safe_num(df[COL_PRODUCED])).abs()

def _find_csv(pattern_base: str) -> Path:
    candidates = sorted(DATA_DIR.glob(f"{pattern_base}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for pattern {pattern_base} in {DATA_DIR}")
    for c in candidates:
        if c.stem == pattern_base:
            return c
    return candidates[0]

def _boxplot(metric: str, data: dict[str, pd.DataFrame], out_name: str, ylabel: str, title: str):
    plt.figure(figsize=(8, 4))
    labels = list(data.keys())
    series = [pd.to_numeric(data[k][metric], errors="coerce").dropna().values for k in labels]
    if not any(len(s) for s in series):
        print(f"[WARN] No data for {metric} boxplot, skipping.")
        plt.close()
        return
    plt.boxplot(series, tick_labels=labels, showfliers=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()

def _bar(values: list[float], labels: list[str], *, title: str, ylabel: str, out_name: str):
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, values, width=0.55)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_f1_box(data: dict[str, pd.DataFrame]):
    for df in data.values():
        _require_cols(df, [COL_F1], "f1")
    _boxplot(COL_F1, data, "B_f1_boxplot.png", "F1 score", "F1 distribution by aggregation config")

def plot_recall_box(data: dict[str, pd.DataFrame]):
    for df in data.values():
        _require_cols(df, [COL_REC], "recall")
    _boxplot(COL_REC, data, "B_recall_boxplot.png", "Recall", "Recall distribution by aggregation config")

def plot_precision_box(data: dict[str, pd.DataFrame]):
    for df in data.values():
        _require_cols(df, [COL_PREC], "precision")
    _boxplot(COL_PREC, data, "B_precision_boxplot.png", "Precision", "Precision distribution by aggregation config")

def plot_desired_vs_produced(data: dict[str, pd.DataFrame]):
    means_desired = []
    means_prod = []
    labels = list(data.keys())
    for name, df in data.items():
        _require_cols(df, [COL_ITEMS, COL_PRODUCED], name)
        means_desired.append(float(_safe_num(df[COL_ITEMS]).mean()))
        means_prod.append(float(_safe_num(df[COL_PRODUCED]).mean()))
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, means_desired, width, label="Desired (items_total)")
    plt.bar(x + width / 2, means_prod, width, label="Produced (chunks)")
    plt.xticks(x, labels)
    plt.xlabel("Configuration")
    plt.ylabel("Mean chunks per run")
    plt.title("Desired vs produced chunk counts per configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "B_desired_vs_produced.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_runtime(data: dict[str, pd.DataFrame]):
    labels = list(data.keys())
    vals = []
    for name, df in data.items():
        if COL_RUNTIME not in df.columns:
            raise ValueError(f"{name} missing {COL_RUNTIME}")
        vals.append(float(_safe_num(df[COL_RUNTIME]).mean()))
    _bar(vals, labels, title="Runtime by configuration", ylabel="Mean chunk_time_sec", out_name="B_runtime_bar.png")

def plot_chunk_diff(data: dict[str, pd.DataFrame]):
    labels = list(data.keys())
    vals = []
    for name, df in data.items():
        vals.append(float(_compute_chunk_diff(df).dropna().mean()))
    _bar(vals, labels, title="Absolute chunk count deviation |desired - produced|", ylabel="Chunks", out_name="B_chunk_diff_bar.png")

print("Running in:", DATA_DIR)
data: dict[str, pd.DataFrame] = {}
pretty_map = {
    "B1_1PDFs": "1 PDF",
    "B2_2PDFs": "2 PDFs",
    "B3_4PDFs": "4 PDFs",
}
for label, pat in CONFIG_PATTERNS.items():
    csv_path = _find_csv(pat)
    pretty_label = pretty_map.get(label, label)
    print(f"{pretty_label}: {csv_path.name}")
    df = pd.read_csv(csv_path)
    data[pretty_label] = df
plot_f1_box(data)
plot_recall_box(data)
plot_precision_box(data)
plot_desired_vs_produced(data)
plot_runtime(data)
plot_chunk_diff(data)
print("Done. Outputs:", OUT_DIR.resolve())
