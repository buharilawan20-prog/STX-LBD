import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# SETTINGS
# =========================
DPI = 300
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.jpg", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=DPI, bbox_inches="tight")


# =========================
# MODEL PERFORMANCE DATA
# =========================
models = [
    "Dino-only\n(Pre-2016 → Post-2015)",
    "Cyano-only\n(Transfer)",
    "Combined\n(Cyano + Dino)"
]

precision = [
    0.2539,   # your temporal validation overall precision
    0.0360,   # cyano → dino transfer
    0.0263    # combined model
]


# =========================
# PLOT
# =========================
fig = plt.figure(figsize=(8, 5))

bars = plt.bar(models, precision)

plt.ylabel("Precision")
plt.title("Model Comparison: Saxitoxin Knowledge Prediction")

# Add value labels
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.005, f"{y:.3f}", ha='center')

plt.ylim(0, max(precision) + 0.05)

save(fig, "Figure8_model_comparison")

plt.show()
