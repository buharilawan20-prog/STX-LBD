#!/bin/bash

BASE="/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT"

OUT="$BASE/RSTUDIO_FIGURE_WORKSPACE"

mkdir -p "$OUT"

echo "======================================="
echo "Preparing RStudio Figure Workspace"
echo "======================================="

# ==========================================
# CORPUS FILES
# ==========================================

mkdir -p "$OUT/corpus"

cp "$BASE/FINAL_WORKSPACE/corpus/dino_pre2016.csv" \
"$OUT/corpus/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/corpus/dino_post2015.csv" \
"$OUT/corpus/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/corpus/cyano_all.csv" \
"$OUT/corpus/" 2>/dev/null

# ==========================================
# ENTITY FILES
# ==========================================

mkdir -p "$OUT/entities"

cp "$BASE/FINAL_WORKSPACE/processed/all_split_entities_combined.csv" \
"$OUT/entities/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/all_split_entities_combined_normalized.csv" \
"$OUT/entities/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/entity_normalization_summary.csv" \
"$OUT/entities/" 2>/dev/null

# ==========================================
# EDGE FILES
# ==========================================

mkdir -p "$OUT/edges"

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_edges.csv" \
"$OUT/edges/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_post2015_edges.csv" \
"$OUT/edges/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/cyano_all_edges.csv" \
"$OUT/edges/" 2>/dev/null

# ==========================================
# HYPOTHESES
# ==========================================

mkdir -p "$OUT/hypotheses"

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_enriched_hypotheses.csv" \
"$OUT/hypotheses/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses.csv" \
"$OUT/hypotheses/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv" \
"$OUT/hypotheses/" 2>/dev/null

# ==========================================
# NODE2VEC
# ==========================================

mkdir -p "$OUT/node2vec"

cp "$BASE/FINAL_WORKSPACE/embeddings/dino_pre2016_node2vec_embeddings.csv" \
"$OUT/node2vec/" 2>/dev/null

# ==========================================
# ML / TEMPORAL VALIDATION
# ==========================================

mkdir -p "$OUT/ml"

cp "$BASE/FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/strict_temporal_validated_hypotheses.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/strict_temporal_validation_metrics.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/node2vec_vs_ai_comparison_metrics.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/ai_ranker_training_summary.csv" \
"$OUT/ml/" 2>/dev/null

# ==========================================
# CROSS TAXA
# ==========================================

mkdir -p "$OUT/cross_taxa"

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cross_taxa_transfer_summary.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cross_taxa_transfer_candidate_summary.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_all_vs_dino_all_conserved_edges.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_all_vs_dino_post2015_convergent_edges.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_plus_dino_pre2016_predicts_dino_post2015.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_cyano_only_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_environment_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_evolutionary_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_gene_related_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/true_divergent_vs_conserved_category_counts.csv" \
"$OUT/cross_taxa/" 2>/dev/null

# ==========================================
# FIGURE FILES
# ==========================================

mkdir -p "$OUT/figures"

cp "$BASE/FINAL_WORKSPACE/figures/"*.png \
"$OUT/figures/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/figures/"*.pdf \
"$OUT/figures/" 2>/dev/null

# ==========================================
# SUMMARY TABLE
# ==========================================

echo ""
echo "======================================="
echo "Workspace Ready"
echo "======================================="
echo ""

find "$OUT" -type f | sort

echo ""
echo "Saved to:"
echo "$OUT"#!/bin/bash

BASE="/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT"

OUT="$BASE/RSTUDIO_FIGURE_WORKSPACE"

mkdir -p "$OUT"

echo "======================================="
echo "Preparing RStudio Figure Workspace"
echo "======================================="

# ==========================================
# CORPUS FILES
# ==========================================

mkdir -p "$OUT/corpus"

cp "$BASE/FINAL_WORKSPACE/corpus/dino_pre2016.csv" \
"$OUT/corpus/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/corpus/dino_post2015.csv" \
"$OUT/corpus/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/corpus/cyano_all.csv" \
"$OUT/corpus/" 2>/dev/null

# ==========================================
# ENTITY FILES
# ==========================================

mkdir -p "$OUT/entities"

cp "$BASE/FINAL_WORKSPACE/processed/all_split_entities_combined.csv" \
"$OUT/entities/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/all_split_entities_combined_normalized.csv" \
"$OUT/entities/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/entity_normalization_summary.csv" \
"$OUT/entities/" 2>/dev/null

# ==========================================
# EDGE FILES
# ==========================================

mkdir -p "$OUT/edges"

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_edges.csv" \
"$OUT/edges/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_post2015_edges.csv" \
"$OUT/edges/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/cyano_all_edges.csv" \
"$OUT/edges/" 2>/dev/null

# ==========================================
# HYPOTHESES
# ==========================================

mkdir -p "$OUT/hypotheses"

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_enriched_hypotheses.csv" \
"$OUT/hypotheses/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses.csv" \
"$OUT/hypotheses/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv" \
"$OUT/hypotheses/" 2>/dev/null

# ==========================================
# NODE2VEC
# ==========================================

mkdir -p "$OUT/node2vec"

cp "$BASE/FINAL_WORKSPACE/embeddings/dino_pre2016_node2vec_embeddings.csv" \
"$OUT/node2vec/" 2>/dev/null

# ==========================================
# ML / TEMPORAL VALIDATION
# ==========================================

mkdir -p "$OUT/ml"

cp "$BASE/FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/strict_temporal_validated_hypotheses.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/strict_temporal_validation_metrics.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/node2vec_vs_ai_comparison_metrics.csv" \
"$OUT/ml/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/ml/ai_ranker_training_summary.csv" \
"$OUT/ml/" 2>/dev/null

# ==========================================
# CROSS TAXA
# ==========================================

mkdir -p "$OUT/cross_taxa"

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cross_taxa_transfer_summary.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cross_taxa_transfer_candidate_summary.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_all_vs_dino_all_conserved_edges.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_all_vs_dino_post2015_convergent_edges.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/cyano_plus_dino_pre2016_predicts_dino_post2015.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_cyano_only_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_environment_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_evolutionary_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/top_gene_related_transfer_candidates.csv" \
"$OUT/cross_taxa/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/cross_taxa/true_divergent_vs_conserved_category_counts.csv" \
"$OUT/cross_taxa/" 2>/dev/null

# ==========================================
# FIGURE FILES
# ==========================================

mkdir -p "$OUT/figures"

cp "$BASE/FINAL_WORKSPACE/figures/"*.png \
"$OUT/figures/" 2>/dev/null

cp "$BASE/FINAL_WORKSPACE/figures/"*.pdf \
"$OUT/figures/" 2>/dev/null

# ==========================================
# SUMMARY TABLE
# ==========================================

echo ""
echo "======================================="
echo "Workspace Ready"
echo "======================================="
echo ""

find "$OUT" -type f | sort

echo ""
echo "Saved to:"
echo "$OUT"
