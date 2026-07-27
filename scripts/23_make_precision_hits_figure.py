import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = (
    BASE /
    "FINAL_WORKSPACE/strict_validation_v2/strict_global_ml_results.csv"
)

OUTDIR = BASE / "FINAL_WORKSPACE/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure_precision_hits_at_k.png"
OUT_PDF = OUTDIR / "Figure_precision_hits_at_k.pdf"

# ======================================================
# LOAD
# ======================================================

df = pd.read_csv(INFILE)

# ======================================================
# SETTINGS
# ======================================================

K_VALUES = [10, 20, 50, 100, 200]

MODELS = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "ExtraTrees",
    "SVM"
]

LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "ExtraTrees": "Extra Trees",
    "SVM": "SVM"
}

COLORS = {
    "LogisticRegression": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "GradientBoosting": "#2ca02c",
    "ExtraTrees": "#d62728",
    "SVM": "#9467bd"
}

df = df[df["Model"].isin(MODELS)]

plt.rcParams["font.family"] = "DejaVu Sans"

# ======================================================
# FIGURE
# ======================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.8)
)

# ======================================================
# PANEL A — PRECISION@K
# ======================================================

ax = axes[0]

for model in MODELS:

    row = df[df["Model"] == model].iloc[0]

    vals = [
        row[f"Precision@{k}"]
        for k in K_VALUES
    ]

    ax.plot(
        K_VALUES,
        vals,
        marker="o",
        linewidth=2.7,
        markersize=7,
        label=LABELS[model],
        color=COLORS[model]
    )

ax.set_title(
    "A. Temporal Precision@K",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Top-ranked hypotheses (K)",
    fontsize=12
)

ax.set_ylabel(
    "Precision@K",
    fontsize=12
)

ax.set_ylim(0, 1.02)

ax.grid(
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    fontsize=10
)

# ======================================================
# PANEL B — HITS@K
# ======================================================

ax = axes[1]

for model in MODELS:

    row = df[df["Model"] == model].iloc[0]

    vals = [
        row[f"Hits@{k}"]
        for k in K_VALUES
    ]

    ax.plot(
        K_VALUES,
        vals,
        marker="s",
        linewidth=2.7,
        markersize=7,
        label=LABELS[model],
        color=COLORS[model]
    )

ax.set_title(
    "B. Future relationships recovered",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Top-ranked hypotheses (K)",
    fontsize=12
)

ax.set_ylabel(
    "Hits@K",
    fontsize=12
)

ax.grid(
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    fontsize=10
)

# ======================================================
# GLOBAL TITLE
# ======================================================

fig.suptitle(
    "Strict global temporal validation of STX-LBD hypothesis ranking",
    fontsize=20,
    fontweight="bold",
    y=1.02
)

fig.text(
    0.5,
    -0.02,
    (
        "Performance was evaluated by testing pre-2016 AI-ranked "
        "dinoflagellate STX hypotheses against post-2015 literature."
    ),
    ha="center",
    fontsize=11
)

plt.tight_layout()

# ======================================================
# SAVE
# ======================================================

plt.savefig(
    OUT_PNG,
    dpi=500,
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
