import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "06_cross_taxa_transfer_categories.png"
OUT_PDF = OUT_DIR / "06_cross_taxa_transfer_categories.pdf"

# ==========================================
# Cross-taxa transfer categories
# ==========================================

data = pd.DataFrame({
    "Category": [
        "cyano_and_dino_prior_signal",
        "cyano_only_prior_signal",
        "new_post2015_only",
        "dino_prior_only_signal"
    ],
    "Count": [
        923,
        345,
        279,
        185
    ]
})

plt.figure(figsize=(10,6))

bars = plt.bar(
    data["Category"],
    data["Count"]
)

plt.title(
    "Cross-taxa transfer categories",
    fontsize=18,
    pad=12
)

plt.ylabel(
    "Number of post-2015 dino edges",
    fontsize=14
)

plt.xticks(
    rotation=30,
    ha="right",
    fontsize=12
)

plt.yticks(fontsize=12)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 10,
        f"{int(height)}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

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

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
