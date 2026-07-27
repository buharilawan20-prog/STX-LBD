import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

# ==========================================
# INPUT FILES
# ==========================================

CONSERVE_FILE = BASE / (
    "FINAL_WORKSPACE/cross_taxa/"
    "cyano_all_vs_dino_post2015_convergent_edges.csv"
)

DIVERGENT_FILE = BASE / (
    "FINAL_WORKSPACE/cross_taxa/"
    "top_cyano_only_transfer_candidates.csv"
)

OUT_DIR = BASE / "FINAL_WORKSPACE/cross_taxa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "true_divergent_vs_conserved_category_counts.csv"

# ==========================================
# CATEGORY RULES
# ==========================================

ENV_TERMS = {
    "temperature",
    "salinity",
    "light",
    "nutrient",
    "nitrogen",
    "nitrate",
    "phosphorus",
    "phosphate",
    "warming",
    "bloom",
    "environment",
    "climate"
}

GENE_TERMS = {
    "sxta",
    "sxtg",
    "sxtd",
    "sxti",
    "sxt",
    "gene",
    "genes",
    "expression",
    "transcriptome",
    "transcription"
}

EVOLUTION_TERMS = {
    "evolution",
    "phylogeny",
    "phylogenetic",
    "divergence",
    "conserved",
    "transfer",
    "adaptation"
}

MECHANISTIC_TERMS = {
    "biosynthesis",
    "toxin_biosynthesis",
    "saxitoxin_biosynthesis",
    "toxin_production",
    "regulation",
    "metabolism",
    "metabolic",
    "pathway",
    "mechanism",
    "functional"
}

# ==========================================
# NORMALIZE
# ==========================================

def normalize(x):

    x = str(x).lower().strip()

    x = x.replace(" ", "_")

    replacements = {
        "sxta1": "sxta",
        "sxta4": "sxta",
        "pst": "paralytic_shellfish_toxins",
        "psts": "paralytic_shellfish_toxins"
    }

    return replacements.get(x, x)

# ==========================================
# CATEGORY ASSIGNMENT
# ==========================================

def assign_category(source, target):

    s = normalize(source)
    t = normalize(target)

    combined = f"{s} {t}"

    # ENVIRONMENTAL
    for term in ENV_TERMS:
        if term in combined:
            return "Environmental"

    # GENE
    for term in GENE_TERMS:
        if term in combined:
            return "Gene-related"

    # EVOLUTION
    for term in EVOLUTION_TERMS:
        if term in combined:
            return "Evolutionary"

    # MECHANISTIC
    for term in MECHANISTIC_TERMS:
        if term in combined:
            return "Mechanistic"

    return "Other"

# ==========================================
# LOAD DATA
# ==========================================

cons_df = pd.read_csv(CONSERVE_FILE).fillna("")
div_df = pd.read_csv(DIVERGENT_FILE).fillna("")

# ==========================================
# PROCESS CONSERVED
# ==========================================

cons_counts = {}

for _, row in cons_df.iterrows():

    s = row.get("source_cyano", "")
    t = row.get("target_cyano", "")

    category = assign_category(s, t)

    cons_counts[category] = cons_counts.get(category, 0) + 1

# ==========================================
# PROCESS DIVERGENT
# ==========================================

div_counts = {}

for _, row in div_df.iterrows():

    s = row.get("source", "")
    t = row.get("target", "")

    category = assign_category(s, t)

    div_counts[category] = div_counts.get(category, 0) + 1

# ==========================================
# MERGE RESULTS
# ==========================================

all_categories = sorted(
    set(cons_counts.keys()) | set(div_counts.keys())
)

rows = []

for cat in all_categories:

    conserved = cons_counts.get(cat, 0)
    divergent = div_counts.get(cat, 0)

    total = conserved + divergent

    if total > 0:
        conserved_pct = conserved / total * 100
        divergent_pct = divergent / total * 100
    else:
        conserved_pct = 0
        divergent_pct = 0

    rows.append({
        "Category": cat,
        "Conserved_Count": conserved,
        "Divergent_Count": divergent,
        "Conserved_Percent": round(conserved_pct, 2),
        "Divergent_Percent": round(divergent_pct, 2)
    })

summary_df = pd.DataFrame(rows)

summary_df = summary_df.sort_values(
    "Conserved_Count",
    ascending=False
)

# ==========================================
# SAVE
# ==========================================

summary_df.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# PRINT
# ==========================================

print("\n========== TRUE CATEGORY COUNTS ==========\n")

print(summary_df.to_string(index=False))

print("\nSaved:")
print(OUT_FILE)
