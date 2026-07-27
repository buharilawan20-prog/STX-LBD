import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "FINAL_WORKSPACE/cross_taxa/true_divergent_vs_conserved_category_counts.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_GLOBAL_100_conserved_vs_divergent_STX_semantic_biology.png"
OUT_PDF = OUT_DIR / "Figure_GLOBAL_100_conserved_vs_divergent_STX_semantic_biology.pdf"

df = pd.read_csv(INPUT)

# Keep only biological categories
order = ["Environmental", "Evolutionary", "Gene-related", "Mechanistic"]

df = df[df["Category"].isin(order)].copy()

df["Category"] = pd.Categorical(
    df["Category"],
    categories=order,
    ordered=True
)

df = df.sort_values("Category")

# ==========================================================
# GLOBAL 100% NORMALIZATION
# ==========================================================

grand_total = df["Conserved_Count"].sum() + df["Divergent_Count"].sum()

df["Conserved_Global_Percent"] = df["Conserved_Count"] / grand_total * 100
df["Divergent_Global_Percent"] = df["Divergent_Count"] / grand_total * 100

# Long format
plot_df = pd.concat([
    df[["Category", "Conserved_Count", "Conserved_Global_Percent"]].rename(
        columns={
            "Conserved_Count": "Count",
            "Conserved_Global_Percent": "Percent"
        }
    ).assign(Relationship_Class="Conserved / transferred"),

    df[["Category", "Divergent_Count", "Divergent_Global_Percent"]].rename(
        columns={
            "Divergent_Count": "Count",
            "Divergent_Global_Percent": "Percent"
        }
    ).assign(Relationship_Class="Divergent / cyano-only")
])

plot_df["Relationship_Class"] = pd.Categorical(
    plot_df["Relationship_Class"],
    categories=["Divergent / cyano-only", "Conserved / transferred"],
    ordered=True
)

# ==========================================================
# PLOT
# ==========================================================

x = np.arange(len(order))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 8))

colors = {
    "Divergent / cyano-only": "#1f77b4",
    "Conserved / transferred": "#ff7f0e"
}

for i, cls in enumerate(["Divergent / cyano-only", "Conserved / transferred"]):

    sub = plot_df[plot_df["Relationship_Class"] == cls].sort_values("Category")

    offset = -width / 2 if i == 0 else width / 2

    bars = ax.bar(
        x + offset,
        sub["Percent"],
        width,
        label=cls,
        color=colors[cls],
        edgecolor="black",
        linewidth=1.1
    )

    for bar, pct, count in zip(bars, sub["Percent"], sub["Count"]):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{pct:.1f}%\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

# ==========================================================
# AXES
# ==========================================================

ax.set_xticks(x)
ax.set_xticklabels(
    order,
    rotation=25,
    ha="right",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Percentage of all relationships",
    fontsize=17,
    fontweight="bold"
)

ax.set_xlabel(
    "Biological category",
    fontsize=17,
    fontweight="bold"
)

ax.set_title(
    "Global distribution of conserved and divergent STX semantic biology",
    fontsize=23,
    fontweight="bold",
    pad=18
)

ax.legend(
    title="Relationship class",
    fontsize=13,
    title_fontsize=15,
    frameon=True
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_ylim(0, max(plot_df["Percent"]) + 8)

ax.text(
    -0.10,
    1.04,
    "A",
    transform=ax.transAxes,
    fontsize=24,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nGrand total relationships:", grand_total)
print(plot_df.sort_values(["Category", "Relationship_Class"]).to_string(index=False))
