import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_enriched_hypotheses.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "dino_pre2016_priority_hypotheses.csv"
REMOVED_OUT = OUT_DIR / "dino_pre2016_low_priority_hypotheses.csv"

PRIORITY_CLASSES = [
    "environment_gene_regulation",
    "taxon_gene_association",
    "cyano_gene_transfer_signal",
    "gene_process_association",
    "environment_toxin_association",
    "process_toxin_association",
    "cross_taxa_association",
    "taxon_toxin_association"
]

LOW_VALUE_TERMS = [
    "toxicity",
    "expression",
    "bloom",
    "dinoflagellate",
    "dinoflagellates",
    "cyanobacteria",
    "cyanobacterial",
    "saxitoxin",
    "paralytic_shellfish_toxins",
    "paralytic_shellfish_poisoning"
]

df = pd.read_csv(INFILE).fillna("")

priority = df[df["Hypothesis_Class"].isin(PRIORITY_CLASSES)].copy()

# Remove very generic source-target pairs
priority["source_generic"] = priority["Source"].isin(LOW_VALUE_TERMS)
priority["target_generic"] = priority["Target"].isin(LOW_VALUE_TERMS)

priority_filtered = priority[
    ~(priority["source_generic"] & priority["target_generic"])
].copy()

priority_filtered = priority_filtered.drop(
    columns=["source_generic", "target_generic"]
)

removed = df[
    ~df.index.isin(priority_filtered.index)
].copy()

priority_filtered = priority_filtered.sort_values(
    by="Score",
    ascending=False
)

priority_filtered.to_csv(
    OUTFILE,
    index=False,
    encoding="utf-8-sig"
)

removed.to_csv(
    REMOVED_OUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== PRIORITY HYPOTHESIS FILTER ==========")
print("Original hypotheses:", len(df))
print("Priority hypotheses retained:", len(priority_filtered))
print("Low-priority removed:", len(removed))

print("\nPriority class distribution:")
print(priority_filtered["Hypothesis_Class"].value_counts())

print("\nSaved:")
print(OUTFILE)
print(REMOVED_OUT)

print("\nTop priority hypotheses:")
print(
    priority_filtered[
        [
            "Source",
            "Source_Type",
            "Target",
            "Target_Type",
            "Hypothesis_Class",
            "Score",
            "Common_Neighbors",
            "Bridge_Nodes"
        ]
    ].head(30).to_string(index=False)
)
