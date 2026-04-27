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

## Repository structure

```text
scripts/      Python scripts for the full workflow
data/         Processed corpus files
results/      Model outputs, validation tables, transfer analyses
figures/      Publication-ready figures
docs/         Additional documentation
