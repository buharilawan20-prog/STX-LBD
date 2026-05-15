# STX-LBD: AI-driven literature-based discovery for saxitoxin biology

This repository contains a normalized semantic knowledge graph and AI-based literature discovery pipeline for saxitoxin (STX) research.

## Overview

The workflow integrates:

1. Literature-derived semantic entities
2. Strict biological normalization
3. Semantic knowledge graph construction
4. Hypothesis generation
5. Node2Vec graph embeddings
6. Machine-learning hypothesis ranking
7. Temporal validation
8. Cross-taxa cyanobacteria–dinoflagellate transfer analysis

## Main workflow

```text
Literature corpus
→ entity and phrase extraction
→ strict normalization
→ semantic edge construction
→ collapsed knowledge graph
→ hypothesis generation
→ Node2Vec embeddings
→ ML ranking
→ temporal validation
→ cross-taxa transfer validation
pip install -r requirements.txt
 
