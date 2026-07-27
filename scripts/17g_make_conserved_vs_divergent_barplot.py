import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "FINAL_WORKSPACE/cross_taxa/cross_taxa_transfer_candidate_summary.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_conserved_vs_divergent_STX_biology_refined.png"
OUT_PDF = OUT_DIR / "Figure_conserved_vs_divergent_STX_biology_refined.pdf"

# ==========================================
# YOUR ACTUAL RESULTS
# ==========================================

# Conserved / transferred
conserved = {
    "Environmental": 574,
    "Evolutionary": 809,
    "Gene-related": 482,
    "Mechanistic": 574
}

# Divergent / cyano-only
divergent = {
    "Environmental": 345,
    "Evolutionary": 279,
    "Gene-related": 482,
    "Mechanistic": 574
}

# ==========================================
# CONVERT TO PERCENTAGES
# ==========================================

categories = list(conserved.keys())

conserved_vals = np.array([conserved[c] for c in categories], dtype=float)
divergent_vals = np.array([divergent[c] for c in categories], dtype=float)

totals = conserved_vals + divergent_vals

conserved_pct = conserved_vals / totals * 100
divergent_pct = divergent_vals / totals * 100

# ==========================================
# PLOT
# ==========================================

x = np.arange(len(categories))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 7))

bars1 = ax.bar(
    x - width/2,
    divergent_pct,
    width,
    label="Divergent / cyano-only",
    edgecolor="black",
    linewidth=1.0
)

bars2 = ax.bar(
    x + width/2,
    conserved_pct,
    width,
    label="Conserved / transferred",
    edgecolor="black",
    linewidth=1.0
)

# ==========================================
# LABELS
# ==========================================

for bar in list(bars1) + list(bars2):

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 1.5,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# ==========================================
# AXES
# ==========================================

ax.set_ylabel(
    "Percentage of relationships",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Biological category",
    fontsize=16,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(
    categories,
    rotation=25,
    fontsize=13,
    fontweight="bold"
)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylim(0, 100)

# ==========================================
# TITLE
# ==========================================

ax.set_title(
    "Conserved versus divergent STX semantic biology",
    fontsize=22,
    fontweight="bold",
    pad=18
)

# ==========================================
# LEGEND
# ==========================================

legend = ax.legend(
    title="Relationship class",
    fontsize=13,
    title_fontsize=15,
    frameon=True
)

legend.get_frame().set_linewidth(1.2)

# ==========================================
# STYLE
# ==========================================

ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
