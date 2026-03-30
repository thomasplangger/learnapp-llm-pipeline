import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load CSV
df = pd.read_csv("lo_grouping_compare_summary_all.csv")

metrics = {
    "Silhouette": "silhouette_cosine",
    "Group Balance": "gini_group_sizes",
    "Coherence": "topic_similarity"
}

methods = df["method"].dropna().unique()

# compute mean values
data = {m: [] for m in methods}

for label, col in metrics.items():
    for m in methods:
        data[m].append(df[df["method"] == m][col].mean())

labels = list(metrics.keys())
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8,4))

for i, m in enumerate(methods):
    ax.bar(x + i*width, data[m], width, label=m)

ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_title("LO Quality Metrics by Grouping Method")
ax.legend()

plt.tight_layout()
plt.show()