import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/splits/dino_pre2016.csv"
INFILE2 = BASE / "FINAL_WORKSPACE/splits/dino_post2015.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_dinoflagellate_corpus_year_distribution.png"
OUT_PDF = OUT_DIR / "Figure_dinoflagellate_corpus_year_distribution.pdf"

# ===============================
# LOAD DINO CORPUS
# ===============================

pre = pd.read_csv(INFILE).fillna("")
post = pd.read_csv(INFILE2).fillna("")

df = pd.concat([pre, post], ignore_index=True)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

# ===============================
# COUNT PAPERS PER YEAR
# ===============================

year_counts = (
    df.groupby("year")
    .size()
    .reset_index(name="papers")
    .sort_values("year")
)

# Fill missing years so the timeline is continuous
all_years = pd.DataFrame({
    "year": range(year_counts["year"].min(), year_counts["year"].max() + 1)
})

year_counts = all_years.merge(year_counts, on="year", how="left").fillna(0)
year_counts["papers"] = year_counts["papers"].astype(int)

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(13, 8))

plt.bar(
    year_counts["year"],
    year_counts["papers"],
    width=0.75
)

# 2015 cutoff line
plt.axvline(
    x=2015,
    linestyle="--",
    linewidth=2.5
)

plt.title(
    "Dinoflagellate STX Corpus Distribution with 2015 Cutoff",
    fontsize=22,
    pad=15
)

plt.xlabel("Year", fontsize=18)
plt.ylabel("Number of Papers", fontsize=18)

plt.xticks(
    year_counts["year"],
    rotation=90,
    fontsize=13
)

plt.yticks(fontsize=14)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nSummary:")
print("Total dinoflagellate records:", len(df))
print("Pre-2016:", len(pre))
print("Post-2015:", len(post))
print("\nYear counts:")
print(year_counts.to_string(index=False))
