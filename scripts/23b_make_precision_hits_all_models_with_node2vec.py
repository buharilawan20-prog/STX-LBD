import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

STRICT_FILE = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_ml_results.csv"
NODE2VEC_FILE = BASE / "FINAL_WORKSPACE/ml/node2vec_vs_ai_comparison_metrics.csv"

OUTDIR = BASE / "FINAL_WORKSPACE/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure_precision_hits_all_models_with_node2vec.png"
OUT_PDF = OUTDIR / "Figure_precision_hits_all_models_with_node2vec.pdf"

K_VALUES = [10, 20, 50, 100, 200]

# ======================================================
# LOAD SUPERVISED STRICT ML RESULTS
# ======================================================

strict = pd.read_csv(STRICT_FILE)

supervised_models = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "ExtraTrees",
    "SVM",
    "MLP"
]

model_labels = {
    "Node2Vec": "Unsupervised Node2Vec",
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "ExtraTrees": "Extra Trees",
    "SVM": "SVM",
    "MLP": "MLP"
}

plot_rows = []

for _, r in strict.iterrows():
    model = r["Model"]

    if model not in supervised_models:
        continue

    for k in K_VALUES:
        plot_rows.append({
            "Model": model,
            "K": k,
            "Precision@K": r[f"Precision@{k}"],
            "Hits@K": r[f"Hits@{k}"],
            "Model_Type": "Supervised ML"
        })

# ======================================================
# LOAD NODE2VEC RESULTS
# ======================================================

node = pd.read_csv(NODE2VEC_FILE)

node = node[node["Method"].astype(str).str.contains("Node2Vec", case=False, na=False)].copy()

for _, r in node.iterrows():
    k = int(r["K"])

    if k not in K_VALUES:
        continue

    plot_rows.append({
        "Model": "Node2Vec",
        "K": k,
        "Precision@K": r["Precision@K"],
        "Hits@K": r["Hits@K"],
        "Model_Type": "Unsupervised"
    })

plot_df = pd.DataFrame(plot_rows)

# ======================================================
# PLOT SETTINGS
# ======================================================

colors = {
    "Node2Vec": "#000000",
    "LogisticRegression": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "GradientBoosting": "#2ca02c",
    "ExtraTrees": "#d62728",
    "SVM": "#9467bd",
    "MLP": "#8c564b"
}

linestyles = {
    "Node2Vec": "--",
    "LogisticRegression": "-",
    "RandomForest": "-",
    "GradientBoosting": "-",
    "ExtraTrees": "-",
    "SVM": "-",
    "MLP": "-"
}

markers = {
    "Node2Vec": "D",
    "LogisticRegression": "o",
    "RandomForest": "s",
    "GradientBoosting": "^",
    "ExtraTrees": "v",
    "SVM": "P",
    "MLP": "X"
}

model_order = [
    "Node2Vec",
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "ExtraTrees",
    "SVM",
    "MLP"
]

plt.rcParams["font.family"] = "DejaVu Sans"

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ======================================================
# PANEL A: PRECISION@K
# ======================================================

ax = axes[0]

for model in model_order:
    g = plot_df[plot_df["Model"] == model].sort_values("K")

    if g.empty:
        continue

    ax.plot(
        g["K"],
        g["Precision@K"],
        marker=markers[model],
        linestyle=linestyles[model],
        color=colors[model],
        linewidth=2.6,
        markersize=7,
        label=model_labels[model]
    )

ax.set_title("A. Temporal Precision@K", fontsize=16, fontweight="bold")
ax.set_xlabel("Top-ranked hypotheses (K)", fontsize=12)
ax.set_ylabel("Precision@K", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(frameon=False, fontsize=9)

# ======================================================
# PANEL B: HITS@K
# ======================================================

ax = axes[1]

for model in model_order:
    g = plot_df[plot_df["Model"] == model].sort_values("K")

    if g.empty:
        continue

    ax.plot(
        g["K"],
        g["Hits@K"],
        marker=markers[model],
        linestyle=linestyles[model],
        color=colors[model],
        linewidth=2.6,
        markersize=7,
        label=model_labels[model]
    )

ax.set_title("B. Future relationships recovered", fontsize=16, fontweight="bold")
ax.set_xlabel("Top-ranked hypotheses (K)", fontsize=12)
ax.set_ylabel("Hits@K", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(frameon=False, fontsize=9)

# ======================================================
# GLOBAL TITLE
# ======================================================

fig.suptitle(
    "Temporal validation performance of STX-LBD hypothesis ranking",
    fontsize=20,
    fontweight="bold",
    y=1.03
)

fig.text(
    0.5,
    -0.03,
    (
        "Node2Vec represents unsupervised ranking; supervised ML models use graph-derived "
        "features to prioritize future post-2015 STX relationships."
    ),
    ha="center",
    fontsize=11
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=500, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nData plotted:")
print(plot_df.to_string(index=False))
