import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "E_merge_split_analysis"
OUT_DIR.mkdir(exist_ok=True)
BASELINE_KEY = "summary_E1_Baseline"
MERGE_10_KEY = "summary_E2_Merge_10"
MERGE_KEY = "summary_E2_Merge_12"
COL_ITEMS = "items_total"
COL_PRODUCED = "produced_chunks"
COL_F1 = "f1"
COL_REC = "recall"
COL_TOO_SMALL = "too_small"
COL_TOO_BIG = "too_big"
COL_CHUNK_DIFF = "chunk_diff"
MIN_PRODUCED_CHUNKS = 3                                       

def _find_csv_by_key(key: str) -> Path:
    candidates = sorted(DATA_DIR.glob(f"{key}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for '{key}' in {DATA_DIR}")
    for c in candidates:
        if c.stem == key:
            return c
    return candidates[0]

def _require_cols(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing columns: {missing}. Have: {list(df.columns)}")

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _drop_small_produced(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if COL_PRODUCED not in df.columns:
        return df
    produced = _safe_num(df[COL_PRODUCED]).fillna(0)
    before = len(df)
    out = df[produced >= MIN_PRODUCED_CHUNKS].copy()
    dropped = before - len(out)
    if dropped:
        print(f"[CLEAN] Dropped {dropped} runs from {label} (produced_chunks < {MIN_PRODUCED_CHUNKS})")
    return out

def _compute_chunk_diff(df: pd.DataFrame) -> pd.Series:
    if COL_CHUNK_DIFF in df.columns:
        return _safe_num(df[COL_CHUNK_DIFF])
    if COL_ITEMS in df.columns and COL_PRODUCED in df.columns:
        return (_safe_num(df[COL_ITEMS]) - _safe_num(df[COL_PRODUCED])).abs()
    raise ValueError("Need either chunk_diff or (items_total and produced_chunks) to compute chunk deviation.")

def _bar_by_config(series_by_config: dict[str, list[float]], xlabels: list[str], *, title: str, ylabel: str, out_name: str):
    config_names = list(series_by_config.keys())
    if not config_names:
        return
    values = [series_by_config[name] for name in config_names]
    if any(len(v) != len(xlabels) for v in values):
        raise ValueError("Each config series must have the same number of values as xlabels.")
    x = np.arange(len(xlabels))
    n = len(config_names)
    width = min(0.8 / max(1, n), 0.28)
    offsets = (np.arange(n) - (n - 1) / 2) * width
    plt.figure(figsize=(8.5, 4))
    for i, name in enumerate(config_names):
        plt.bar(x + offsets[i], values[i], width, label=name)
    plt.xticks(x, xlabels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()

def _boxplot_by_config(
    vals_by_config: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    out_name: str,

):
    labels = list(vals_by_config.keys())
    data = [pd.to_numeric(vals_by_config[k], errors="coerce").dropna().values for k in labels]
    if not any(len(d) for d in data):
        print(f"[WARN] No data for {out_name}")
        return
    plt.figure(figsize=(8.5, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()

def _scatter_by_config(
    vals_by_config: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    out_name: str,
):
    labels = list(vals_by_config.keys())
    xs = np.arange(len(labels))
    colors = {
        "baseline": "black",
        "merge_10": "#1f77b4",
        "merge_12": "orange",
    }
    plt.figure(figsize=(8.5, 4))
    for i, name in enumerate(labels):
        s = pd.to_numeric(vals_by_config[name], errors="coerce").dropna()
        if s.empty:
            continue
        jitter = np.random.normal(0.0, 0.06, size=len(s))
        plt.scatter(
            np.full(len(s), xs[i]) + jitter,
            s.values,
            s=28,
            alpha=0.65,
            color=colors.get(name, None),
            label=name,
            zorder=2,
        )
    plt.xticks(xs, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_merge_split_chunk_diff(dfs: dict[str, pd.DataFrame]):
    """Average absolute deviation |produced - desired| across configurations."""
    vals: dict[str, list[float]] = {}
    for name, df in dfs.items():
        s = _compute_chunk_diff(df).dropna()
        vals[name] = [float(s.mean() if not s.empty else 0.0)]
    _bar_by_config(
        vals,
        xlabels=["Avg |produced - desired|"],
        title="Average deviation between produced and reference chunk counts",
        ylabel="Chunks (absolute deviation)",
        out_name="merge_split_chunk_diff.png",
    )

def plot_merge_split_size_violations(dfs: dict[str, pd.DataFrame]):
    """Average number of undersized and oversized chunks per run across configurations."""
    vals: dict[str, list[float]] = {}
    for name, df in dfs.items():
        _require_cols(df, [COL_TOO_SMALL, COL_TOO_BIG], name)
        vals[name] = [
            float(_safe_num(df[COL_TOO_SMALL]).mean()),
            float(_safe_num(df[COL_TOO_BIG]).mean()),
        ]
    _bar_by_config(
        vals,
        xlabels=["Too small", "Too big"],
        title="Average size-window violations per run",
        ylabel="Mean chunks per run",
        out_name="merge_split_size_violations.png",
    )

def plot_f1_recall_scatter_and_boxplots(dfs: dict[str, pd.DataFrame]):
    """Per-run scatter + boxplots for F1 and recall across configurations."""
    f1_vals: dict[str, pd.Series] = {}
    rec_vals: dict[str, pd.Series] = {}
    for name, df in dfs.items():
        _require_cols(df, [COL_F1, COL_REC], name)
        f1_vals[name] = _safe_num(df[COL_F1])
        rec_vals[name] = _safe_num(df[COL_REC])
    _scatter_by_config(
        f1_vals,
        title="F1 per run by configuration",
        ylabel="F1",
        out_name="merge_split_f1_scatter.png",
    )
    _boxplot_by_config(
        f1_vals,
        title="F1 distribution by configuration",
        ylabel="F1",
        out_name="merge_split_f1_boxplot.png",
    )
    _scatter_by_config(
        rec_vals,
        title="Recall per run by configuration",
        ylabel="Recall",
        out_name="merge_split_recall_scatter.png",
    )
    _boxplot_by_config(
        rec_vals,
        title="Recall distribution by configuration",
        ylabel="Recall",
        out_name="merge_split_recall_boxplot.png",
    )

def plot_desired_vs_produced(dfs: dict[str, pd.DataFrame]):
    """Plot like analyze_scaling_A desired vs produced, but with conditions = configs."""
    desired: list[float] = []
    produced: list[float] = []
    labels: list[str] = []
    for name, df in dfs.items():
        _require_cols(df, [COL_ITEMS, COL_PRODUCED], name)
        labels.append(name)
        desired.append(float(_safe_num(df[COL_ITEMS]).mean()))
        produced.append(float(_safe_num(df[COL_PRODUCED]).mean()))
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, desired, width, label="Desired (items_total)")
    plt.bar(x + width / 2, produced, width, label="Produced (chunks)")
    plt.xticks(x, labels)
    plt.xlabel("Configuration")
    plt.ylabel("Mean chunks per run")
    plt.title("Desired vs produced chunks by condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "merge_split_desired_vs_produced.png", dpi=300, bbox_inches="tight")
    plt.close()
print("Running in:", DATA_DIR)
baseline_csv = _find_csv_by_key(BASELINE_KEY)
merge_10_csv = _find_csv_by_key(MERGE_10_KEY)
merge_12_csv = _find_csv_by_key(MERGE_KEY)
print("Baseline:", baseline_csv.name)
print("Merge 10:", merge_10_csv.name)
print("Merge 12:", merge_12_csv.name)
df_base = pd.read_csv(baseline_csv)
df_m10 = pd.read_csv(merge_10_csv)
df_m12 = pd.read_csv(merge_12_csv)
dfs = {
    "baseline": _drop_small_produced(df_base, label="baseline"),
    "merge_10": _drop_small_produced(df_m10, label="merge_10"),
    "merge_12": _drop_small_produced(df_m12, label="merge_12"),
}
plot_merge_split_chunk_diff(dfs)
plot_merge_split_size_violations(dfs)
plot_f1_recall_scatter_and_boxplots(dfs)
plot_desired_vs_produced(dfs)
print("Done. Outputs:", OUT_DIR.resolve())
