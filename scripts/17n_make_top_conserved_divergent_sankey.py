import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_top_conserved_divergent_STX_relationships.png"
OUT_PDF = OUT_DIR / "Figure_top_conserved_divergent_STX_relationships.pdf"

# ==========================================================
# CURATED TOP RELATIONSHIPS
# Replace/add based on your final table if needed
# ==========================================================

data = [
    # category, relationship, weight, outcome
    ("Gene–toxin relationships", "biosynthesis ↔ STX", 57, "Conserved"),
    ("Gene–toxin relationships", "STX ↔ sxtA", 35, "Conserved"),
    ("Gene–toxin relationships", "biosynthesis ↔ PSTs", 28, "Conserved"),
    ("Gene–toxin relationships", "STX ↔ toxin production", 23, "Conserved"),
    ("Environmental regulation", "nitrogen ↔ STX", 21, "Conserved"),

    ("Gene–toxin relationships", "neoSTX ↔ sxtA", 5, "Divergent"),
    ("Biosynthetic mechanisms", "arginine ↔ biosynthesis", 4, "Divergent"),
    ("Environmental regulation", "biosynthesis ↔ climate change", 4, "Divergent"),
    ("Environmental regulation", "phosphate ↔ phosphorus", 4, "Divergent"),
    ("Biosynthetic mechanisms", "biosynthetic pathway ↔ STX biosynthesis", 3, "Divergent"),
]

df = pd.DataFrame(
    data,
    columns=["Category", "Relationship", "Weight", "Outcome"]
)

# ==========================================================
# SETTINGS
# ==========================================================

category_order = [
    "Gene–toxin relationships",
    "Environmental regulation",
    "Biosynthetic mechanisms"
]

outcome_order = ["Conserved", "Divergent"]

cat_colors = {
    "Gene–toxin relationships": "#2A9D8F",
    "Environmental regulation": "#E76F51",
    "Biosynthetic mechanisms": "#6A4C93"
}

outcome_colors = {
    "Conserved": "#2A9D8F",
    "Divergent": "#E76F51"
}

flow_colors = {
    "Conserved": "#9BD3C9",
    "Divergent": "#F3A08D"
}

# ==========================================================
# LAYOUT
# ==========================================================

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

x_cat = 0.08
x_rel = 0.50
x_out = 0.86

cat_w = 0.19
out_w = 0.14
rel_w = 0.25

# Relationship vertical positions
df = df.sort_values(["Outcome", "Weight"], ascending=[True, False]).reset_index(drop=True)

rel_y_positions = {}
top_y = 0.82
gap = 0.075

for i, row in df.iterrows():
    rel_y_positions[row["Relationship"]] = top_y - i * gap

# Category box heights based on count
cat_counts = df["Category"].value_counts().to_dict()
cat_y_start = {
    "Gene–toxin relationships": 0.48,
    "Environmental regulation": 0.30,
    "Biosynthetic mechanisms": 0.18
}

cat_heights = {
    "Gene–toxin relationships": 0.34,
    "Environmental regulation": 0.13,
    "Biosynthetic mechanisms": 0.10
}

# Outcome boxes
out_y = {
    "Conserved": 0.46,
    "Divergent": 0.22
}
out_h = {
    "Conserved": 0.36,
    "Divergent": 0.13
}

# ==========================================================
# DRAW HELPERS
# ==========================================================

def draw_box(x, y, w, h, text, color, fontsize=12):
    rect = Rectangle(
        (x, y), w, h,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.95
    )
    ax.add_patch(rect)
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

def draw_rel_box(x, y, text, weight):
    h = 0.055
    rect = Rectangle(
        (x - rel_w/2, y - h/2),
        rel_w,
        h,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        f"{text}\nW={weight}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

def draw_flow(x0, y0, x1, y1, width, color, alpha=0.72):
    verts = [
        (x0, y0),
        ((x0 + x1) / 2, y0),
        ((x0 + x1) / 2, y1),
        (x1, y1)
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4
    ]
    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=width,
        alpha=alpha,
        capstyle="round"
    )
    ax.add_patch(patch)

# ==========================================================
# TITLES
# ==========================================================

ax.text(
    0.02,
    0.965,
    "Cross-taxa conservation and divergence of saxitoxin relationships",
    fontsize=22,
    fontweight="bold",
    ha="left"
)

ax.text(
    0.02,
    0.925,
    "Top cyanobacteria-derived STX relationships evaluated across the dinoflagellate corpus",
    fontsize=14,
    ha="left",
    color="#444444"
)

ax.text(x_cat, 0.88, "Biological category", fontsize=15, fontweight="bold", ha="left")
ax.text(x_rel, 0.88, "Representative relationship", fontsize=15, fontweight="bold", ha="center")
ax.text(x_out, 0.88, "Cross-taxa outcome", fontsize=15, fontweight="bold", ha="center")

# ==========================================================
# CATEGORY BOXES
# ==========================================================

for cat in category_order:
    n = cat_counts.get(cat, 0)
    draw_box(
        x_cat,
        cat_y_start[cat],
        cat_w,
        cat_heights[cat],
        f"{cat}\n(n={n})",
        cat_colors[cat],
        fontsize=12
    )

# ==========================================================
# OUTCOME BOXES
# ==========================================================

for outcome in outcome_order:
    n = (df["Outcome"] == outcome).sum()
    draw_box(
        x_out,
        out_y[outcome],
        out_w,
        out_h[outcome],
        f"{outcome}\n(n={n})",
        outcome_colors[outcome],
        fontsize=15
    )

# ==========================================================
# RELATIONSHIP BOXES AND FLOWS
# ==========================================================

max_w = df["Weight"].max()

# assign category connection y positions
cat_anchor_counter = {cat: 0 for cat in category_order}
cat_anchor_step = {
    cat: cat_heights[cat] / max(cat_counts.get(cat, 1), 1)
    for cat in category_order
}

out_anchor_counter = {out: 0 for out in outcome_order}
out_anchor_step = {
    out: out_h[out] / max((df["Outcome"] == out).sum(), 1)
    for out in outcome_order
}

for _, row in df.iterrows():

    cat = row["Category"]
    rel = row["Relationship"]
    weight = row["Weight"]
    outcome = row["Outcome"]

    y_rel = rel_y_positions[rel]

    draw_rel_box(x_rel, y_rel, rel, weight)

    # Category anchor
    cy = (
        cat_y_start[cat]
        + cat_heights[cat]
        - (cat_anchor_counter[cat] + 0.5) * cat_anchor_step[cat]
    )
    cat_anchor_counter[cat] += 1

    # Outcome anchor
    oy = (
        out_y[outcome]
        + out_h[outcome]
        - (out_anchor_counter[outcome] + 0.5) * out_anchor_step[outcome]
    )
    out_anchor_counter[outcome] += 1

    lw = 2.0 + (weight / max_w) * 14

    draw_flow(
        x_cat + cat_w,
        cy,
        x_rel - rel_w/2,
        y_rel,
        lw,
        flow_colors[outcome],
        alpha=0.65
    )

    draw_flow(
        x_rel + rel_w/2,
        y_rel,
        x_out,
        oy,
        lw,
        flow_colors[outcome],
        alpha=0.75
    )

# ==========================================================
# FOOTNOTE
# ==========================================================

ax.text(
    0.02,
    0.06,
    "W = relationship weight in cyanobacterial STX literature. Conserved = detected in dinoflagellate literature; Divergent = absent or weakly supported in dinoflagellate literature.",
    fontsize=11,
    color="#444444"
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nRelationships used:")
print(df.to_string(index=False))
