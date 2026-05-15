#!/bin/bash

echo "Step 1: Prepare corpus"
python scripts/01_prepare_corpus.py

echo "Step 2: Extract entities"
python scripts/02_extract_entities.py

echo "Step 3: Normalize entities"
python scripts/03_normalize_entities.py

echo "Step 4: Build edges"
python scripts/04_build_edges.py

echo "Step 5: Build graph"
python scripts/05_build_graph.py

echo "Step 6: Generate hypotheses"
python scripts/06_generate_hypotheses.py

echo "Step 7: Train embeddings"
python scripts/07_train_node2vec.py

echo "Step 8: ML scoring"
python scripts/15_ml_score_discovery_shortlist.py

echo "Step 9: Validation"
python scripts/14_validate_discovery_hypotheses.py

echo "Done!"
