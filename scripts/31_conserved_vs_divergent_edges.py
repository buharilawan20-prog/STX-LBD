import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")
OUTDIR = BASE / "FINAL_WORKSPACE/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "05_conserved_vs_divergent_edges.png"
OUT_PDF = OUTDIR / "05_conserved_vs_divergent_edges.pdf"

data = pd.DataFrame({
    "Category": ["Conserved", "Dino-specific", "Cyano-specific"],
    "Edges": [1481, 697, 366]
})

plt.figure(figsize=(10, 6))

bars = plt.bar(data["Category"], data["Edges"])

plt.title(
    "Conserved vs divergent cross-taxa STX semantic edges",
    fontsize=18,
    pad=12
)

plt.ylabel("Number of semantic edges", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 25,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=13
    )

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=500, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
