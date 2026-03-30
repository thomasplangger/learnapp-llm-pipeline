import pandas as pd
from scipy.stats import ttest_rel

# ---- load file ----
df = pd.read_csv("lo_grouping_compare_summary_all_5Page.csv")

# ---- normalize and filter methods ----
# convert to string, strip whitespace, lowercase
df["method"] = df["method"].astype(str).str.strip().str.lower()

# some rows may be 'nan' as a string now, drop those + anything unexpected
valid_methods = {"llm", "heuristic"}
df = df[df["method"].isin(valid_methods)].copy()

print("Detected method labels after cleaning:", df["method"].unique())

# sanity check
assert set(df["method"].unique()) == {"llm", "heuristic"}, "Unexpected method labels after cleaning"

# ---- list of metrics to test ----
metrics = [
    "mean_chunks_per_group",
    "median_chunks_per_group",
    "std_chunks_per_group",
    "mean_intra_cosine",
    "mean_inter_cosine",
    "gini_group_sizes",
    "max_group_share",
    "silhouette_cosine",
    "topic_similarity",
    "keyword_similarity",
    "mean_summary_chunk_sim",
    "summary_margin",
]

results = []

for m in metrics:
    # extract paired values
    llm_vals = df[df["method"] == "llm"][m].values
    heur_vals = df[df["method"] == "heuristic"][m].values

    # safety check: make sure lengths match
    if len(llm_vals) != len(heur_vals):
        print(f"Skipping metric {m}: unequal lengths (llm={len(llm_vals)}, heuristic={len(heur_vals)})")
        continue

    # paired t-test
    t, p = ttest_rel(llm_vals, heur_vals)

    mean_diff = (llm_vals - heur_vals).mean()

    results.append({
        "metric": m,
        "mean_diff (LLM−heuristic)": mean_diff,
        "p_value": p,
    })

res_df = pd.DataFrame(results).sort_values("p_value")

print("\n=== Paired t-tests (LLM vs heuristic) ===\n")
print(res_df.to_string(index=False))
