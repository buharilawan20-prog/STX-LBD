import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

AI_FILE = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"
NODE2VEC_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_TOP_OUT = OUT_DIR / "top_100_ai_ranked_hypotheses_for_interpretation.csv"
NODE2VEC_TOP_OUT = OUT_DIR / "top_100_node2vec_hypotheses_for_interpretation.csv"
CLASS_SUMMARY_OUT = OUT_DIR / "top_hypotheses_class_summary.csv"

KEEP_COLS = [
    "Source",
    "Source_Type",
    "Target",
    "Target_Type",
    "Hypothesis_Class",
    "Temporal_Label",
    "ML_Probability",
    "Node2Vec_Integrated_Score",
    "Final_AI_Rank_Score",
    "Score",
    "Common_Neighbors",
    "Distinct_Bridge_Types",
    "Bridge_Nodes"
]

def clean_export(df):
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[KEEP_COLS].copy()

    df["Interpretation"] = df.apply(make_interpretation, axis=1)

    return df

def make_interpretation(row):
    source = row["Source"]
    target = row["Target"]
    hclass = row["Hypothesis_Class"]
    bridges = str(row["Bridge_Nodes"]).split(";")[:5]
    bridges = [b.strip() for b in bridges if b.strip()]
    bridge_text = ", ".join(bridges)

    if hclass == "environment_gene_regulation":
        return f"{source} may influence or regulate {target}, supported by bridge concepts including {bridge_text}."

    elif hclass == "taxon_gene_association":
        return f"{source} may be associated with {target}-related genetic capacity, supported by bridge concepts including {bridge_text}."

    elif hclass == "cyano_gene_transfer_signal":
        return f"{source} may provide cross-taxa context for {target}, suggesting possible evolutionary or biosynthetic linkage supported by {bridge_text}."

    elif hclass == "gene_process_association":
        return f"{source} may be linked to the biological process {target}, supported by bridge concepts including {bridge_text}."

    elif hclass == "process_toxin_association":
        return f"{source} may be mechanistically associated with {target}, supported by bridge concepts including {bridge_text}."

    elif hclass == "environment_toxin_association":
        return f"{source} may be associated with variation in {target}, supported by bridge concepts including {bridge_text}."

    elif hclass == "taxon_toxin_association":
        return f"{source} may be associated with {target} production or occurrence, supported by bridge concepts including {bridge_text}."

    elif hclass == "cross_taxa_association":
        return f"{source} and {target} may share STX-related biological context, supported by bridge concepts including {bridge_text}."

    else:
        return f"{source} may be associated with {target}, supported by bridge concepts including {bridge_text}."

ai_df = pd.read_csv(AI_FILE).fillna("")
node2vec_df = pd.read_csv(NODE2VEC_FILE).fillna("")

ai_top = ai_df.sort_values(
    by="Final_AI_Rank_Score",
    ascending=False
).head(100)

node_top = node2vec_df.sort_values(
    by="Node2Vec_Integrated_Score",
    ascending=False
).head(100)

ai_export = clean_export(ai_top)
node_export = clean_export(node_top)

ai_export.to_csv(AI_TOP_OUT, index=False, encoding="utf-8-sig")
node_export.to_csv(NODE2VEC_TOP_OUT, index=False, encoding="utf-8-sig")

class_summary = pd.DataFrame({
    "AI_Top100": ai_export["Hypothesis_Class"].value_counts(),
    "Node2Vec_Top100": node_export["Hypothesis_Class"].value_counts()
}).fillna(0).astype(int)

class_summary.to_csv(CLASS_SUMMARY_OUT, encoding="utf-8-sig")

print("\n========== EXPORT TOP HYPOTHESES ==========")

print("\nSaved AI top 100:")
print(AI_TOP_OUT)

print("\nSaved Node2Vec top 100:")
print(NODE2VEC_TOP_OUT)

print("\nSaved class summary:")
print(CLASS_SUMMARY_OUT)

print("\nAI Top 20:")
print(
    ai_export[
        [
            "Source",
            "Target",
            "Hypothesis_Class",
            "Temporal_Label",
            "ML_Probability",
            "Final_AI_Rank_Score",
            "Interpretation"
        ]
    ].head(20).to_string(index=False)
)
