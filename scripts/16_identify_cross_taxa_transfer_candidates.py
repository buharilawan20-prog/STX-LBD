import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/cross_taxa/cyano_plus_dino_pre2016_predicts_dino_post2015.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/cross_taxa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CYANO_ONLY_OUT = OUT_DIR / "top_cyano_only_transfer_candidates.csv"
GENE_TRANSFER_OUT = OUT_DIR / "top_gene_related_transfer_candidates.csv"
ENV_TRANSFER_OUT = OUT_DIR / "top_environment_transfer_candidates.csv"
EVOLUTION_TRANSFER_OUT = OUT_DIR / "top_evolutionary_transfer_candidates.csv"
SUMMARY_OUT = OUT_DIR / "cross_taxa_transfer_candidate_summary.csv"

df = pd.read_csv(INFILE).fillna("")

df["transfer_support_score"] = pd.to_numeric(
    df["transfer_support_score"],
    errors="coerce"
).fillna(0)

df["dino_post_weight"] = pd.to_numeric(
    df["dino_post_weight"],
    errors="coerce"
).fillna(0)

df["cyano_prior_weight"] = pd.to_numeric(
    df["cyano_prior_weight"],
    errors="coerce"
).fillna(0)

# ===============================
# CYANO-ONLY PRIOR SIGNALS
# ===============================

cyano_only = df[
    df["transfer_type"] == "cyano_only_prior_signal"
].copy()

cyano_only = cyano_only.sort_values(
    by=["cyano_prior_weight", "dino_post_weight"],
    ascending=False
)

cyano_only.to_csv(
    CYANO_ONLY_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# GENE-RELATED TRANSFER SIGNALS
# ===============================

gene_related = df[
    (
        df["source_type"].astype(str).str.contains("SXT_GENE", case=False, na=False) |
        df["target_type"].astype(str).str.contains("SXT_GENE", case=False, na=False) |
        df["source"].astype(str).str.contains("sxt|sxta|biosynthesis", case=False, na=False) |
        df["target"].astype(str).str.contains("sxt|sxta|biosynthesis", case=False, na=False)
    )
].copy()

gene_related = gene_related.sort_values(
    by=["transfer_support_score", "dino_post_weight"],
    ascending=False
)

gene_related.to_csv(
    GENE_TRANSFER_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# ENVIRONMENT-RELATED TRANSFER SIGNALS
# ===============================

ENV_TERMS = [
    "temperature",
    "salinity",
    "nutrient",
    "nitrogen",
    "phosphorus",
    "phosphate",
    "nitrate",
    "light",
    "climate",
    "warming",
    "bloom",
    "stress"
]

env_pattern = "|".join(ENV_TERMS)

environment_related = df[
    (
        df["source_type"].astype(str).str.contains("ENV_FACTOR", case=False, na=False) |
        df["target_type"].astype(str).str.contains("ENV_FACTOR", case=False, na=False) |
        df["source"].astype(str).str.contains(env_pattern, case=False, na=False) |
        df["target"].astype(str).str.contains(env_pattern, case=False, na=False)
    )
].copy()

environment_related = environment_related.sort_values(
    by=["transfer_support_score", "dino_post_weight"],
    ascending=False
)

environment_related.to_csv(
    ENV_TRANSFER_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# EVOLUTION-RELATED TRANSFER SIGNALS
# ===============================

EVO_TERMS = [
    "evolution",
    "phylogeny",
    "phylogenetic",
    "horizontal_gene_transfer",
    "gene_loss",
    "gene_duplication",
    "functional_divergence",
    "biosynthesis",
    "sxta"
]

evo_pattern = "|".join(EVO_TERMS)

evolution_related = df[
    (
        df["source"].astype(str).str.contains(evo_pattern, case=False, na=False) |
        df["target"].astype(str).str.contains(evo_pattern, case=False, na=False) |
        df["post2015_relation"].astype(str).str.contains("gene|process|biosynthesis", case=False, na=False)
    )
].copy()

evolution_related = evolution_related.sort_values(
    by=["transfer_support_score", "dino_post_weight"],
    ascending=False
)

evolution_related.to_csv(
    EVOLUTION_TRANSFER_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# SUMMARY
# ===============================

summary = pd.DataFrame({
    "category": [
        "all_post2015_edges",
        "cyano_only_prior_signal",
        "gene_related_transfer",
        "environment_related_transfer",
        "evolution_related_transfer"
    ],
    "count": [
        len(df),
        len(cyano_only),
        len(gene_related),
        len(environment_related),
        len(evolution_related)
    ]
})

summary.to_csv(
    SUMMARY_OUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== CROSS-TAXA TRANSFER CANDIDATES ==========")
print(summary.to_string(index=False))

print("\nSaved:")
print(CYANO_ONLY_OUT)
print(GENE_TRANSFER_OUT)
print(ENV_TRANSFER_OUT)
print(EVOLUTION_TRANSFER_OUT)
print(SUMMARY_OUT)

print("\nTop cyano-only transfer candidates:")
print(
    cyano_only[
        [
            "source",
            "target",
            "post2015_relation",
            "dino_post_weight",
            "cyano_prior_weight",
            "transfer_type"
        ]
    ].head(30).to_string(index=False)
)

print("\nTop gene-related transfer candidates:")
print(
    gene_related[
        [
            "source",
            "target",
            "post2015_relation",
            "dino_post_weight",
            "cyano_prior_weight",
            "dino_pre_prior_weight",
            "transfer_type"
        ]
    ].head(30).to_string(index=False)
)
