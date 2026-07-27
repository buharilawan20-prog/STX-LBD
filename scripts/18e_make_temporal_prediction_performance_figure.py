import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

ML = BASE / "FINAL_WORKSPACE/ml"
OUT = BASE / "FINAL_WORKSPACE/figures"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = ML / "node2vec_vs_ai_comparison_metrics.csv"

OUT_PNG = OUT / "Figure_temporal_prediction_performance.png"
OUT_PDF = OUT / "Figure_temporal_prediction_performance.pdf"

df = pd.read_csv(INPUT)

df = df[df["K"].isin([10, 20, 50, 100, 200])].copy()
df = df.sort_values(["Method", "K"])

method_labels = {
    "Node2Vec": "Unsupervised Node2Vec",
    "AI_Ranker": "Supervised AI ranker"
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ======================================================
# Panel A: Precision@K
# ======================================================

ax = axes[0]

for method, g in df.groupby("Method"):
    ax.plot(
        g["K"],
        g["Precision@K"],
        marker="o",
        linewidth=2.8,
        markersize=8,
        label=method_labels.get(method, method)
    )

ax.set_title(
    "A. Temporal precision",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Top-ranked hypotheses (K)", fontsize=12)
ax.set_ylabel("Precision@K", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(frameon=False)

# ======================================================
# Panel B: Hits@K
# ======================================================

ax = axes[1]

for method, g in df.groupby("Method"):
    ax.plot(
        g["K"],
        g["Hits@K"],
        marker="s",
        linewidth=2.8,
        markersize=8,
        label=method_labels.get(method, method)
    )

ax.set_title(
    "B. Future relationships recovered",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Top-ranked hypotheses (K)", fontsize=12)
ax.set_ylabel("Hits@K", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(frameon=False)

fig.suptitle(
    "Temporal prediction performance of STX-LBD hypothesis ranking",
    fontsize=18,
    fontweight="bold",
    y=1.03
)

fig.text(
    0.5,
    -0.03,
    "Performance was evaluated by testing pre-2016-ranked hypotheses against post-2015 dinoflagellate STX literature.",
    ha="center",
    fontsize=11
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nData used:")
print(df.to_string(index=False))
