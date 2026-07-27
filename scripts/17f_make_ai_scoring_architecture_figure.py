import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_AI_scoring_architecture_STX_LBD.png"
OUT_PDF = OUT_DIR / "Figure_AI_scoring_architecture_STX_LBD.pdf"

# ===============================
# FIGURE HELPERS
# ===============================

def draw_box(ax, x, y, w, h, text, fc="#F2F2F2", ec="#333333", fontsize=11):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc
    )

    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True
    )

    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=lw,
        color=color
    )

    ax.add_patch(arrow)


# ===============================
# DRAW
# ===============================

fig, ax = plt.subplots(figsize=(15, 9))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Colors
c_input = "#D9EAF7"
c_kg = "#DDEFD8"
c_feature = "#FFF2CC"
c_embed = "#EADCF8"
c_model = "#F8D7DA"
c_output = "#DDEBF7"
c_validation = "#E2F0D9"

# ===============================
# INPUT / KG
# ===============================

draw_box(
    ax, 0.04, 0.68, 0.18, 0.14,
    "Pre-2016\nDinoflagellate STX\nliterature corpus",
    fc=c_input
)

draw_box(
    ax, 0.28, 0.68, 0.18, 0.14,
    "Semantic knowledge graph\n(nodes: toxins, genes,\ntaxa, environment,\nprocesses)",
    fc=c_kg,
    fontsize=10
)

draw_box(
    ax, 0.52, 0.68, 0.18, 0.14,
    "Candidate hypothesis\npair (u, v)\nfrom disconnected\nsemantic nodes",
    fc=c_input,
    fontsize=10
)

draw_arrow(ax, 0.22, 0.75, 0.28, 0.75)
draw_arrow(ax, 0.46, 0.75, 0.52, 0.75)

# ===============================
# FEATURE EXTRACTION LAYER
# ===============================

draw_box(
    ax, 0.12, 0.40, 0.20, 0.15,
    "Structural KG features\n\nCommon neighbors\nBridge score\nAdamic–Adar\nJaccard\nPreferential attachment",
    fc=c_feature,
    fontsize=9
)

draw_box(
    ax, 0.40, 0.40, 0.20, 0.15,
    "Biological semantic signals\n\nEntity types\nHypothesis class\nBridge-node diversity\nGene–toxin–environment\ncontext",
    fc=c_kg,
    fontsize=9
)

draw_box(
    ax, 0.68, 0.40, 0.20, 0.15,
    "Node2Vec embedding features\n\nSource–target cosine\nBridge mean similarity\nBridge max similarity\nIntegrated embedding score",
    fc=c_embed,
    fontsize=9
)

draw_arrow(ax, 0.61, 0.68, 0.22, 0.55)
draw_arrow(ax, 0.61, 0.68, 0.50, 0.55)
draw_arrow(ax, 0.61, 0.68, 0.78, 0.55)

# ===============================
# FEATURE VECTOR
# ===============================

draw_box(
    ax, 0.34, 0.22, 0.32, 0.10,
    "Integrated feature vector\n(structural + semantic + embedding features)",
    fc="#FCE4D6",
    fontsize=11
)

draw_arrow(ax, 0.22, 0.40, 0.42, 0.32)
draw_arrow(ax, 0.50, 0.40, 0.50, 0.32)
draw_arrow(ax, 0.78, 0.40, 0.58, 0.32)

# ===============================
# ML MODEL
# ===============================

draw_box(
    ax, 0.34, 0.07, 0.32, 0.10,
    "Supervised AI ranker\n(Logistic Regression, Random Forest,\nGradient Boosting)",
    fc=c_model,
    fontsize=11
)

draw_arrow(ax, 0.50, 0.22, 0.50, 0.17)

# ===============================
# OUTPUT
# ===============================

draw_box(
    ax, 0.74, 0.13, 0.20, 0.14,
    "Final STX-LBD\nhypothesis score\n\nML probability +\nNode2Vec score",
    fc=c_output,
    fontsize=11
)

draw_arrow(ax, 0.66, 0.12, 0.74, 0.20)

# ===============================
# VALIDATION
# ===============================

draw_box(
    ax, 0.74, 0.68, 0.20, 0.14,
    "Post-2015\nDinoflagellate literature\nused for temporal\nvalidation labels",
    fc=c_validation,
    fontsize=10
)

draw_arrow(ax, 0.84, 0.68, 0.84, 0.27)

ax.text(
    0.875,
    0.48,
    "Temporal validation\n(validated vs unvalidated)",
    ha="center",
    va="center",
    fontsize=10,
    rotation=90,
    fontweight="bold"
)

# ===============================
# TITLE AND NOTES
# ===============================

ax.text(
    0.5,
    0.95,
    "AI scoring architecture for STX literature-based discovery",
    ha="center",
    va="center",
    fontsize=19,
    fontweight="bold"
)

ax.text(
    0.5,
    0.905,
    "Pre-2016 dinoflagellate knowledge is transformed into semantic hypotheses, scored using graph topology,\nNode2Vec embeddings and supervised AI ranking, then evaluated against post-2015 literature.",
    ha="center",
    va="center",
    fontsize=11
)

# Panel label
ax.text(
    0.015,
    0.965,
    "A",
    ha="left",
    va="top",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
