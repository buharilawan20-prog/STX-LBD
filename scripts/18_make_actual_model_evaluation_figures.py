import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")
ML = BASE / "FINAL_WORKSPACE/ml"
CROSS = BASE / "FINAL_WORKSPACE/cross_taxa"
OUT = BASE / "FINAL_WORKSPACE/figures"
OUT.mkdir(parents=True, exist_ok=True)

comparison_file = ML / "node2vec_vs_ai_comparison_metrics.csv"
strict_file = ML / "strict_temporal_validation_metrics.csv"
training_file = ML / "ai_ranker_training_summary.csv"

out_png = OUT / "Figure_actual_model_evaluation_panels.png"
out_pdf = OUT / "Figure_actual_model_evaluation_panels.pdf"

compare = pd.read_csv(comparison_file)
strict = pd.read_csv(strict_file)

# Try loading supervised model summary
try:
    train = pd.read_csv(training_file)
except Exception:
    train = pd.DataFrame()

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(18, 12))

# ==========================================================
# A. Node2Vec vs AI Precision@K
# ==========================================================
ax1 = plt.subplot2grid((2, 2), (0, 0))

for method, g in compare.groupby("Method"):
    g = g.sort_values("K")
    ax1.plot(
        g["K"],
        g["Precision@K"],
        marker="o",
        linewidth=2.5,
        label=method
    )

ax1.set_title("A. Node2Vec vs supervised AI ranking", fontsize=16, fontweight="bold")
ax1.set_xlabel("Top K")
ax1.set_ylabel("Precision@K")
ax1.set_ylim(0, 1.05)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.legend(frameon=False)

# ==========================================================
# B. Hits@K comparison
# ==========================================================
ax2 = plt.subplot2grid((2, 2), (0, 1))

for method, g in compare.groupby("Method"):
    g = g.sort_values("K")
    ax2.plot(
        g["K"],
        g["Hits@K"],
        marker="s",
        linewidth=2.5,
        label=method
    )

ax2.set_title("B. Future discoveries recovered", fontsize=16, fontweight="bold")
ax2.set_xlabel("Top K")
ax2.set_ylabel("Hits@K")
ax2.grid(axis="y", linestyle="--", alpha=0.35)
ax2.legend(frameon=False)

# ==========================================================
# C. Strict temporal validation: Precision and Recall
# ==========================================================
ax3 = plt.subplot2grid((2, 2), (1, 0))

strict = strict.sort_values("K")

ax3.plot(
    strict["K"],
    strict["Precision@K"],
    marker="o",
    linewidth=2.5,
    label="Precision@K"
)

ax3.plot(
    strict["K"],
    strict["Recall@K"],
    marker="^",
    linewidth=2.5,
    label="Recall@K"
)

ax3.set_title("C. Strict temporal validation", fontsize=16, fontweight="bold")
ax3.set_xlabel("Top K")
ax3.set_ylabel("Score")
ax3.set_ylim(0, 1.05)
ax3.grid(axis="y", linestyle="--", alpha=0.35)
ax3.legend(frameon=False)

# ==========================================================
# D. Best top-K comparison barplot
# ==========================================================
ax4 = plt.subplot2grid((2, 2), (1, 1))

topk = compare[compare["K"].isin([10, 20, 50, 100, 200])].copy()

bar_df = topk.pivot_table(
    index="K",
    columns="Method",
    values="Precision@K",
    aggfunc="first"
).reset_index()

methods = [c for c in bar_df.columns if c != "K"]

x = np.arange(len(bar_df["K"]))
width = 0.35

for i, method in enumerate(methods):
    ax4.bar(
        x + (i - 0.5) * width,
        bar_df[method],
        width=width,
        label=method,
        edgecolor="black"
    )

ax4.set_xticks(x)
ax4.set_xticklabels(bar_df["K"])
ax4.set_ylim(0, 1.05)
ax4.set_xlabel("Top K")
ax4.set_ylabel("Precision@K")
ax4.set_title("D. Precision@K by ranking strategy", fontsize=16, fontweight="bold")
ax4.legend(frameon=False)
ax4.grid(axis="y", linestyle="--", alpha=0.35)

plt.suptitle(
    "STX-LBD model evaluation and temporal validation",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(out_png, dpi=400, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(out_png)
print(out_pdf)

print("\nNode2Vec vs AI comparison:")
print(compare.to_string(index=False))

print("\nStrict temporal validation:")
print(strict.to_string(index=False))
