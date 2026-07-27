import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / (
    "FINAL_WORKSPACE/cross_taxa/"
    "true_divergent_vs_conserved_category_counts.csv"
)

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / (
    "Figure_TRUE_conserved_vs_divergent_STX_semantic_biology.png"
)

OUT_PDF = OUT_DIR / (
    "Figure_TRUE_conserved_vs_divergent_STX_semantic_biology.pdf"
)

# ==========================================
# LOAD
# ==========================================

df = pd.read_csv(INPUT)

# Remove Other for cleaner biology-focused figure
df = df[df["Category"] != "Other"].copy()

# Order
order = [
    "Environmental",
    "Evolutionary",
    "Gene-related",
    "Mechanistic"
]

df["Category"] = pd.Categorical(
    df["Category"],
    categories=order,
    ordered=True
)

df = df.sort_values("Category")

# ==========================================
# VALUES
# ==========================================

categories = df["Category"].tolist()

conserved = df["Conserved_Percent"].values
divergent = df["Divergent_Percent"].values

conserved_counts = df["Conserved_Count"].values
divergent_counts = df["Divergent_Count"].values

# ==========================================
# PLOT
# ==========================================

x = np.arange(len(categories))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 8))

bars1 = ax.bar(
    x - width/2,
    divergent,
    width,
    label="Divergent / cyano-only",
    edgecolor="black",
    linewidth=1.2
)

bars2 = ax.bar(
    x + width/2,
    conserved,
    width,
    label="Conserved / transferred",
    edgecolor="black",
    linewidth=1.2
)

# ==========================================
# LABELS
# ==========================================

for i, bar in enumerate(bars1):

    h = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + 1.3,
        f"{h:.1f}%\n(n={divergent_counts[i]})",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

for i, bar in enumerate(bars2):

    h = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + 1.3,
        f"{h:.1f}%\n(n={conserved_counts[i]})",
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
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel(
    "Biological category",
    fontsize=18,
    fontweight="bold"
)

ax.set_xticks(x)

ax.set_xticklabels(
    categories,
    rotation=20,
    fontsize=15,
    fontweight="bold"
)

ax.tick_params(axis="y", labelsize=13)

ax.set_ylim(0, 100)

# ==========================================
# TITLE
# ==========================================

ax.set_title(
    "Conserved versus divergent STX semantic biology",
    fontsize=28,
    fontweight="bold",
    pad=22
)

# ==========================================
# LEGEND
# ==========================================

legend = ax.legend(
    title="Relationship class",
    fontsize=14,
    title_fontsize=17,
    frameon=True
)

legend.get_frame().set_linewidth(1.3)

# ==========================================
# STYLE
# ==========================================

ax.spines["top"].set_linewidth(1.6)
ax.spines["right"].set_linewidth(1.6)
ax.spines["left"].set_linewidth(1.6)
ax.spines["bottom"].set_linewidth(1.6)

# Add subtle grid
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

# Panel label
ax.text(
    -0.12,
    1.03,
    "A",
    transform=ax.transAxes,
    fontsize=24,
    fontweight="bold"
)

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

plt.savefig(
    OUT_PNG,
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
