# 🧬 STX-LBD Explorer

**AI-powered Literature-Based Discovery for Saxitoxin Biosynthesis**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

STX-LBD Explorer is an interactive web application for exploring **AI-prioritized hypotheses** generated through literature-based discovery (LBD) for **saxitoxin (STX) biosynthesis**. The platform integrates semantic knowledge graphs, graph representation learning, machine learning, and temporal validation to identify biologically meaningful relationships among genes, taxa, toxins, environmental factors, and biological processes.

The application accompanies the manuscript:

> **Artificial Intelligence–Driven Literature-Based Discovery Reveals Novel Biological Relationships in Saxitoxin Biosynthesis**  
> *(Manuscript under review)*

---

## Features

- 🔍 **Biological Entity Explorer**
  - Search genes, taxa, toxins, environmental factors, biological processes, and detection methods.
  - Retrieve AI-ranked semantic relationships.
  - Filter by temporal validation status and hypothesis class.

- 🤖 **AI Hypothesis Explorer**
  - Browse all prioritized hypotheses.
  - Rank by AI prediction score.
  - Export filtered results.

- 🕸 **Knowledge Graph Explorer**
  - Interactive visualization of semantic relationships.
  - Explore node connectivity and neighborhood structure.
  - Search specific biological entities.

- 📊 **Statistics & Validation**
  - Corpus statistics
  - Entity distributions
  - Knowledge graph summaries
  - Temporal validation results
  - AI model performance

- 📖 **Documentation**
  - Workflow description
  - Methodology
  - Reproducibility information
  - Citation details

---

# Workflow

The STX-LBD framework consists of six major stages:

```
Literature Collection
        │
        ▼
Semantic Entity Extraction
        │
        ▼
Knowledge Graph Construction
        │
        ▼
Node2Vec Representation Learning
        │
        ▼
Machine Learning Hypothesis Ranking
        │
        ▼
Temporal Validation
        │
        ▼
Interactive STX-LBD Explorer
```

---

# Data Sources

The literature corpus was constructed from multiple scientific databases, including:

- PubMed
- OpenAlex
- CrossRef

The final corpus contains:

| Component | Size |
|------------|------|
| Publications | 1,749 |
| Semantic entity mentions | 9,918 |
| Semantic relationships | 2,178 |
| Priority hypotheses | 514 |
| Future-positive hypotheses | 135 |

---

# Project Structure

```
STX-LBD/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── searchable_hypotheses.csv
│   ├── dino_all_semantic_edges.csv
│   └── figure_ready/
│
├── pages/
│   ├── 1_Entity_Explorer.py
│   ├── 2_AI_Hypotheses.py
│   ├── 3_Knowledge_Graph.py
│   ├── 4_Statistics.py
│   └── 5_Documentation.py
│
├── components/
│
└── assets/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/buharilawan20-prog/STX-LBD.git
cd STX-LBD
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run locally

```bash
streamlit run app.py
```

---

# Deployment

The application is designed for deployment using **Streamlit Community Cloud**.

---

# Scientific Methodology

The framework integrates:

- Literature-based discovery
- Semantic text mining
- Ontology-guided entity normalization
- Semantic knowledge graph construction
- Node2Vec graph embeddings
- Supervised machine learning
- Strict temporal validation
- Cross-taxa knowledge transfer

---

# Repository Contents

| Folder | Description |
|---------|-------------|
| data | Processed datasets used by the application |
| pages | Streamlit interface pages |
| assets | Images and workflow figures |
| components | Shared UI modules |
| app.py | Main application |

---

# Citation

If you use STX-LBD Explorer in your research, please cite:

```
B. L. Muhammad et al.

Artificial Intelligence–Driven Literature-Based Discovery Reveals Novel Biological Relationships in Saxitoxin Biosynthesis.

(Under review)
```

---

# License

This project is released under the MIT License.

---

# Contact

**Dr. Buhari Lawan Muhammad**

Postdoctoral Researcher

Institute of Natural Science

Sangmyung University

Seoul, Republic of Korea

Email: buharilawan20@gmail.com

---

## Acknowledgements

This work was conducted at the Institute of Natural Science, Sangmyung University, Republic of Korea.

The authors acknowledge the use of publicly available scientific literature from PubMed, OpenAlex, and CrossRef in constructing the semantic corpus.
