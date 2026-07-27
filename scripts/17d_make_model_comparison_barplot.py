import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_model_comparison_saxitoxin_prediction.png"
OUT_PDF = OUT_DIR / "Figure_model_comparison_saxitoxin_prediction.pdf"

# ===============================
# VALUES FROM YOUR RESULTS
# ===============================
# Dino-only = strict Node2Vec overall validation: 135 / 514 = 0.263
# Cyano-only transfer = cyano-only prior signal: 345 / 1732 = 0.199
# Combined transfer = cyano + dino prior support: use post2015 edges with either cyano or dino pre prior

values = pd.DataFrame({
    "Model": [
        "Dino-only\n(Pre-2016 → Post-2015)",
        "Cyano-only\n(Transfer)",
        "Combined\n(Cyano + Dino)"
    ],
    "Precision": [
        135 / 514,
        345 / 1732,
        (1268 + 1108 - 923) / 1732  # overlap-adjusted estimate if overlap unknown; replace if needed
    ]
})

# If you prefer conservative combined prior signal from your summary:
# post2015 with cyano prior OR dino prior is approximately:
# cyano prior = 1268, dino prior = 1108, new-only = 279
# therefore supported by at least one prior = 1732 - 279 = 1453
values.loc[2, "Precision"] = 1453 / 1732

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    values["Model"],
    values["Precision"],
    width=0.8
)

plt.title(
    "Model Comparison: Saxitoxin Knowledge Prediction",
    fontsize=20,
    pad=12
)

plt.ylabel("Precision / Recovery fraction", fontsize=16)

plt.ylim(0, max(values["Precision"]) + 0.12)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add values above bars
for bar, val in zip(bars, values["Precision"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=15
    )

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nValues used:")
print(values.to_string(index=False))
