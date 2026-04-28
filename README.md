# STX-LBD: AI-driven saxitoxin knowledge graph and hypothesis discovery

This repository contains scripts and processed outputs for an AI-assisted literature-based discovery framework for saxitoxin (STX) biology.

## Study overview

The pipeline integrates:
- literature corpus preparation
- semantic entity extraction
- n-gram phrase mining
- knowledge graph construction
- hypothesis generation
- node2vec/ML scoring
- temporal validation
- cyanobacteria-to-dinoflagellate transfer analysis
- conserved vs divergent STX biology analysis

## Main experiments

1. Dinoflagellate temporal validation  
   Pre-2016 dinoflagellate literature was used to generate hypotheses, which were validated against post-2015 dinoflagellate literature.

2. Semantic discovery  
   Advanced biological entities and phrase mining were used to generate mechanistic STX hypotheses.

3. Cyanobacteria transfer analysis  
   Cyanobacterial STX literature was used to test cross-taxa transfer to dinoflagellate STX biology.

4. Combined model analysis  
   Cyanobacteria + pre-2016 dinoflagellate knowledge was tested against post-2015 dinoflagellate literature.

## Key conclusion

The dinoflagellate-only model achieved the strongest predictive performance. Cyanobacteria-derived knowledge showed limited but meaningful transfer, indicating a conserved functional STX core but strong lineage-specific mechanisms.

## Key Contributions
Predicts biologically meaningful STX relationships from literature
Demonstrates temporal predictive capability using pre-2016 data
Reveals limited cross-taxa knowledge transfer (~3–4%)
Identifies conserved functional core and divergent mechanisms
Shows that naïve data integration can reduce predictive performance

## Repository Structure

STX-LBD/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── scripts/
│   ├── 01_prepare_corpus.py
│   ├── 02_extract_entities.py
│   ├── 03_normalize_entities.py
│   ├── 04_build_edges.py
│   ├── 05_build_graph.py
│   ├── 06_generate_hypotheses.py
│   ├── 07_train_node2vec.py
│   ├── 14_validate_discovery_hypotheses.py
│   ├── 15_ml_score_discovery_shortlist.py
│   ├── 21_validate_combined_model.py
│
├── results/
│   ├── final_ai_discoveries/
│   ├── temporal_validation/
│   ├── cyano_transfer/
│
├── figures/
│
├── environment.yml
├── run_pipeline.sh
└── README.md

## Reproducibility

1. Clone the repository

git clone https://github.com/buharilawan20-prog/STX-LBD.git
cd STX-LBD

2. Create the environment

conda env create -f environment.yml
conda activate stx_ai

3. Run the pipeline

bash run_pipeline.sh

## Outputs

Running the pipeline generates:

AI-scored hypotheses
Temporal validation results
Cross-taxa transfer analysis
Combined model evaluation
Publication-ready figures (PNG/PDF)

Outputs are saved in:

results/
figures/

## Data

Processed datasets used in this study are included in the repository.

Raw literature data can be retrieved from PubMed using the provided scripts.

## Methods Summary

The workflow consists of:

Literature collection and preprocessing
Entity extraction and semantic enrichment
Knowledge graph construction
Hypothesis generation (link prediction)
Feature extraction (structural + embedding)
Machine learning ranking
Temporal validation
Cross-taxa transfer analysis

## Model Overview

Candidate node pairs are scored using:

Structural graph features (common neighbors, Adamic–Adar, etc.)
Biological signals (STX, environmental drivers, evolution)
Embedding similarity (node2vec)

Final ranking is obtained using machine learning models.

## Significance

This study demonstrates that AI-driven knowledge graph approaches can:

Predict toxin-related biological relationships
Reveal conserved and divergent mechanisms across taxa
Identify limitations in cross-domain knowledge transfer
Support predictive understanding of harmful algal bloom dynamics

## Code Availability

All scripts, processed data, and analysis workflows are publicly available in this repository.

## Status

This repository accompanies a manuscript currently under review.

## Contact

For questions or collaboration:

Buhari Lawan Muhammad PhD
Research professor, SangMyung University
buharilawan20@gmail.com
