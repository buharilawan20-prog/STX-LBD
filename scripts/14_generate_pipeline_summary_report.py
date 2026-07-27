import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

OUT_DIR = BASE / "FINAL_WORKSPACE/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "master_corpus": BASE / "data/processed/stx_enriched_master_corpus_FINAL.csv",
    "dino_pre2016": BASE / "FINAL_WORKSPACE/splits/dino_pre2016.csv",
    "dino_post2015": BASE / "FINAL_WORKSPACE/splits/dino_post2015.csv",
    "cyano_all": BASE / "FINAL_WORKSPACE/splits/cyano_all.csv",
    "entities": BASE / "FINAL_WORKSPACE/processed/all_split_entities_combined_normalized.csv",
    "combined_edges": BASE / "FINAL_WORKSPACE/kg/combined_enriched_semantic_edges.csv",
    "priority_hypotheses": BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses.csv",
    "node2vec_scored": BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv",
    "ai_ranked": BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv",
    "strict_metrics": BASE / "FINAL_WORKSPACE/ml/strict_temporal_validation_metrics.csv",
    "comparison_metrics": BASE / "FINAL_WORKSPACE/ml/node2vec_vs_ai_comparison_metrics.csv",
}

SUMMARY_OUT = OUT_DIR / "stx_lbd_pipeline_summary.csv"
TEXT_OUT = OUT_DIR / "stx_lbd_pipeline_summary.txt"

rows = []

for name, path in FILES.items():
    if path.exists():
        df = pd.read_csv(path).fillna("")
        rows.append({
            "component": name,
            "file": str(path),
            "records": len(df),
            "columns": len(df.columns)
        })
    else:
        rows.append({
            "component": name,
            "file": str(path),
            "records": "MISSING",
            "columns": "MISSING"
        })

summary = pd.DataFrame(rows)
summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

# Load key metrics
strict = pd.read_csv(FILES["strict_metrics"]).fillna("")
comparison = pd.read_csv(FILES["comparison_metrics"]).fillna("")

with open(TEXT_OUT, "w", encoding="utf-8") as f:
    f.write("STX-LBD CORPUS EXPANSION AND AI HYPOTHESIS GENERATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("Pipeline components:\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")

    f.write("Strict temporal validation metrics:\n")
    f.write(strict.to_string(index=False))
    f.write("\n\n")

    f.write("Node2Vec vs supervised AI ranking comparison:\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\n")

    f.write("Suggested interpretation:\n")
    f.write(
        "The enriched STX-LBD framework integrates multidatabase corpus harvesting, "
        "manual corpus recovery, semantic normalization, n-gram phrase mining, "
        "knowledge graph construction, Node2Vec embeddings, temporal validation, "
        "and supervised AI ranking. Strict temporal validation evaluates pre-2016 "
        "predictions against post-2015 literature without additional supervised "
        "training, while the AI ranker represents supervised temporal prioritization.\n"
    )

print("\nSaved summary CSV:")
print(SUMMARY_OUT)

print("\nSaved text report:")
print(TEXT_OUT)

print("\nPipeline summary:")
print(summary.to_string(index=False))
