import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_ml_results.csv"
CANDIDATES = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_candidates.csv"

OUT = BASE / "FINAL_WORKSPACE/figures"
OUT.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT / "Figure_strict_global_temporal_validation_v2.png"
OUT_PDF = OUT / "Figure_strict_global_temporal_validation_v2.pdf"

df = pd.read_csv(INFILE)
cand = pd.read_csv(CANDIDATES)

K_VALUES = [10, 20, 50, 100, 200]

model_order = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "ExtraTrees",
    "SVM",
    "MLP"
]

model_labels = {
    "LogisticRegression": "Logistic\nRegression",
    "RandomForest": "Random\nForest",
    "GradientBoosting": "Gradient\nBoosting",
    "ExtraTrees": "Extra\nTrees",
    "SVM": "SVM",
    "MLP": "MLP"
}

df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df = df.sort_values("Model")

total_candidates = len(cand)
future_pos = int(cand["label"].sum())
future_neg = int((cand["label"] == 0).sum())
positive_rate = future_pos / total_candidates

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(18, 12))

gs = fig.add_gridspec(
    3, 3,
    height_ratios=[0.23, 1, 1],
    width_ratios=[1, 1, 1],
    hspace=0.45,
    wspace=0.32
)

# ======================================================
# TOP SUMMARY STRIP
# ======================================================

ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")

summary_text = (
    f"Strict global temporal validation: pre-2016 KG → post-2015 recovery   |   "
    f"Candidate pairs: {total_candidates:,}   |   "
    f"Future positives: {future_pos:,} ({positive_rate*100:.1f}%)   |   "
    f"Future negatives: {future_neg:,} ({(1-positive_rate)*100:.1f}%)"
)

ax0.text(
    0.5, 0.55,
    summary_text,
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="#F4F6F7",
        edgecolor="black",
        linewidth=1.0
    )
)

# ======================================================
# PANEL A: ROC-AUC
# ======================================================

ax1 = fig.add_subplot(gs[1, 0])

bars = ax1.bar(
    [model_labels[m] for m in df["Model"]],
    df["ROC_AUC"],
    edgecolor="black"
)

ax1.set_title("A. ROC-AUC", fontsize=15, fontweight="bold")
ax1.set_ylabel("ROC-AUC")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

for bar, val in zip(bars, df["ROC_AUC"]):
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# ======================================================
# PANEL B: PR-AUC
# ======================================================

ax2 = fig.add_subplot(gs[1, 1])

bars = ax2.bar(
    [model_labels[m] for m in df["Model"]],
    df["PR_AUC"],
    edgecolor="black"
)

ax2.axhline(
    positive_rate,
    linestyle="--",
    linewidth=1.4,
    color="gray",
    label=f"Random baseline = {positive_rate:.2f}"
)

ax2.set_title("B. PR-AUC", fontsize=15, fontweight="bold")
ax2.set_ylabel("PR-AUC")
ax2.set_ylim(0, max(0.6, df["PR_AUC"].max() + 0.12))
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.legend(frameon=False, fontsize=9)

for bar, val in zip(bars, df["PR_AUC"]):
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.015,
        f"{val:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# ======================================================
# PANEL C: Precision@K
# ======================================================

ax3 = fig.add_subplot(gs[1, 2])

for _, row in df.iterrows():
    model = row["Model"]
    vals = [row[f"Precision@{k}"] for k in K_VALUES]

    ax3.plot(
        K_VALUES,
        vals,
        marker="o",
        linewidth=2.2,
        label=model_labels[model].replace("\n", " ")
    )

ax3.axhline(
    positive_rate,
    linestyle="--",
    linewidth=1.3,
    color="gray",
    label="Random baseline"
)

ax3.set_title("C. Precision@K", fontsize=15, fontweight="bold")
ax3.set_xlabel("Top K")
ax3.set_ylabel("Precision@K")
ax3.set_ylim(0, 1.0)
ax3.grid(axis="y", linestyle="--", alpha=0.3)
ax3.legend(frameon=False, fontsize=8)

# ======================================================
# PANEL D: Hits@K
# ======================================================

ax4 = fig.add_subplot(gs[2, 0])

for _, row in df.iterrows():
    model = row["Model"]
    vals = [row[f"Hits@{k}"] for k in K_VALUES]

    ax4.plot(
        K_VALUES,
        vals,
        marker="s",
        linewidth=2.2,
        label=model_labels[model].replace("\n", " ")
    )

ax4.set_title("D. Future relationships recovered", fontsize=15, fontweight="bold")
ax4.set_xlabel("Top K")
ax4.set_ylabel("Hits@K")
ax4.grid(axis="y", linestyle="--", alpha=0.3)
ax4.legend(frameon=False, fontsize=8)

# ======================================================
# PANEL E: Precision@50 and Hits@50
# ======================================================

ax5 = fig.add_subplot(gs[2, 1])

x = np.arange(len(df))
width = 0.38

p50 = df["Precision@50"].values
h50 = df["Hits@50"].values

b1 = ax5.bar(
    x - width/2,
    p50,
    width,
    label="Precision@50",
    edgecolor="black"
)

ax5.set_ylabel("Precision@50")
ax5.set_ylim(0, 1.0)
ax5.set_xticks(x)
ax5.set_xticklabels([model_labels[m] for m in df["Model"]], fontsize=9)

ax5b = ax5.twinx()

b2 = ax5b.bar(
    x + width/2,
    h50,
    width,
    label="Hits@50",
    edgecolor="black",
    alpha=0.55
)

ax5b.set_ylabel("Hits@50")

ax5.set_title("E. Top-50 recovery", fontsize=15, fontweight="bold")
ax5.grid(axis="y", linestyle="--", alpha=0.3)

for bar, val in zip(b1, p50):
    ax5.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.02,
        f"{val:.2f}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

for bar, val in zip(b2, h50):
    ax5b.text(
        bar.get_x() + bar.get_width()/2,
        val + 1,
        f"{int(val)}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="upper right")

# ======================================================
# PANEL F: Summary table
# ======================================================

ax6 = fig.add_subplot(gs[2, 2])
ax6.axis("off")

table_df = df[[
    "Model", "ROC_AUC", "PR_AUC", "RR",
    "Precision@10", "Precision@50", "Hits@50", "Hits@100"
]].copy()

table_df["Model"] = table_df["Model"].map(lambda x: model_labels[x].replace("\n", " "))

for col in ["ROC_AUC", "PR_AUC", "RR", "Precision@10", "Precision@50"]:
    table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")

for col in ["Hits@50", "Hits@100"]:
    table_df[col] = table_df[col].map(lambda x: f"{int(x)}")

table = ax6.table(
    cellText=table_df.values,
    colLabels=[
        "Model", "ROC", "PR", "RR",
        "P@10", "P@50", "H@50", "H@100"
    ],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.08, 1.55)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.4)

    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#EAECEE")

ax6.set_title("F. Summary metrics", fontsize=15, fontweight="bold", pad=12)

# ======================================================
# GLOBAL TITLE
# ======================================================

fig.suptitle(
    "Strict global temporal validation of STX-LBD hypothesis prediction",
    fontsize=22,
    fontweight="bold",
    y=0.985
)

fig.text(
    0.5,
    0.012,
    "Candidate pairs were generated from the pre-2016 dinoflagellate KG after hub filtering; positives were relationships recovered in the post-2015 KG.",
    ha="center",
    fontsize=11
)

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print("\nInput used:")
print(INFILE)
print(CANDIDATES)
