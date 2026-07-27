import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_overall_STX_LBD_workflow.png"
OUT_PDF = OUT_DIR / "Figure_overall_STX_LBD_workflow.pdf"

def box(ax, x, y, w, h, text, fc, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#333333",
        facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True
    )

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            color="#444444"
        )
    )

fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Colors
c_input = "#D9EAF7"
c_clean = "#E2F0D9"
c_kg = "#DDEFD8"
c_ai = "#EADCF8"
c_val = "#F8D7DA"
c_out = "#FFF2CC"

# Title
ax.text(
    0.5, 0.95,
    "Overall workflow of the STX-LBD framework",
    ha="center", va="center",
    fontsize=22, fontweight="bold"
)

# Row 1: Corpus construction
box(ax, 0.04, 0.76, 0.16, 0.11,
    "Multidatabase\nliterature harvesting\n\nPubMed\nOpenAlex\nCrossRef",
    c_input)

box(ax, 0.25, 0.76, 0.16, 0.11,
    "Corpus cleaning\n\nDeduplication\nRelevance filtering\nManual corpus recovery",
    c_clean)

box(ax, 0.46, 0.76, 0.16, 0.11,
    "Final enriched\nSTX corpus\n\n1,749 records",
    c_out)

box(ax, 0.67, 0.76, 0.16, 0.11,
    "Temporal and taxon split\n\nDino pre-2016\nDino post-2015\nCyano corpus",
    c_input)

arrow(ax, 0.20, 0.815, 0.25, 0.815)
arrow(ax, 0.41, 0.815, 0.46, 0.815)
arrow(ax, 0.62, 0.815, 0.67, 0.815)

# Row 2: Semantic layer
box(ax, 0.10, 0.54, 0.18, 0.12,
    "Semantic extraction\n\nToxins\nsxt genes\nTaxa\nEnvironment\nProcesses",
    c_clean)

box(ax, 0.34, 0.54, 0.18, 0.12,
    "Ontology refinement\nand n-gram mining\n\nPST/PSTs merged\nsxtA variants merged\nsemantic phrases",
    c_clean)

box(ax, 0.58, 0.54, 0.18, 0.12,
    "Semantic knowledge graphs\n\nDino pre-2016 KG\nDino post-2015 KG\nCyano KG",
    c_kg)

arrow(ax, 0.75, 0.76, 0.68, 0.66)
arrow(ax, 0.28, 0.60, 0.34, 0.60)
arrow(ax, 0.52, 0.60, 0.58, 0.60)

# Row 3: Hypothesis + AI
box(ax, 0.05, 0.31, 0.17, 0.12,
    "Hypothesis generation\n\nDisconnected node pairs\nBridge nodes\nStructural graph scores",
    c_out)

box(ax, 0.28, 0.31, 0.17, 0.12,
    "Node2Vec embedding\n\n64-dimensional\nsemantic vectors\ncosine similarity",
    c_ai)

box(ax, 0.51, 0.31, 0.17, 0.12,
    "AI hypothesis ranking\n\nLogistic regression\nRandom forest\nGradient boosting",
    c_ai)

box(ax, 0.74, 0.31, 0.17, 0.12,
    "Final STX-LBD score\n\nGraph features\nEmbedding similarity\nSemantic features",
    c_out)

arrow(ax, 0.67, 0.54, 0.14, 0.43)
arrow(ax, 0.22, 0.37, 0.28, 0.37)
arrow(ax, 0.45, 0.37, 0.51, 0.37)
arrow(ax, 0.68, 0.37, 0.74, 0.37)

# Row 4: Validation + interpretation
box(ax, 0.08, 0.09, 0.18, 0.12,
    "Strict temporal validation\n\nPre-2016 predictions\nvalidated against\npost-2015 literature",
    c_val)

box(ax, 0.32, 0.09, 0.18, 0.12,
    "Cross-taxa transfer\n\nCyano STX knowledge\nvs future dino\nSTX relationships",
    c_val)

box(ax, 0.56, 0.09, 0.18, 0.12,
    "Biological interpretation\n\nValidated hypotheses\nUnvalidated predictions\nBridge mechanisms",
    c_out)

box(ax, 0.80, 0.09, 0.15, 0.12,
    "Outputs\n\nFigures\nTables\nRanked hypotheses\nSTX-LBD framework",
    c_out)

arrow(ax, 0.60, 0.31, 0.17, 0.21)
arrow(ax, 0.83, 0.31, 0.41, 0.21)
arrow(ax, 0.50, 0.15, 0.56, 0.15)
arrow(ax, 0.74, 0.15, 0.80, 0.15)

# Notes
ax.text(
    0.5, 0.015,
    "STX-LBD integrates literature mining, semantic knowledge graphs, graph embeddings, supervised AI ranking, temporal validation, and cross-taxa transfer analysis.",
    ha="center", va="bottom",
    fontsize=11
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
