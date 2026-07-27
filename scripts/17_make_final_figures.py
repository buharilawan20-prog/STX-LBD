import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

FIG_DIR = BASE / "FINAL_WORKSPACE/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR = BASE / "FINAL_WORKSPACE/splits"
KG_DIR = BASE / "FINAL_WORKSPACE/kg"
ML_DIR = BASE / "FINAL_WORKSPACE/ml"
CROSS_DIR = BASE / "FINAL_WORKSPACE/cross_taxa"

# ===============================
# 1. CORPUS TEMPORAL DISTRIBUTION
# ===============================

dino_pre = pd.read_csv(SPLIT_DIR / "dino_pre2016.csv").fillna("")
dino_post = pd.read_csv(SPLIT_DIR / "dino_post2015.csv").fillna("")
cyano = pd.read_csv(SPLIT_DIR / "cyano_all.csv").fillna("")

corpus_counts = pd.DataFrame({
    "Corpus": ["Dino pre-2016", "Dino post-2015", "Cyano all"],
    "Records": [len(dino_pre), len(dino_post), len(cyano)]
})

plt.figure(figsize=(7, 5))
plt.bar(corpus_counts["Corpus"], corpus_counts["Records"])
plt.ylabel("Number of records")
plt.title("Corpus distribution after temporal and taxonomic split")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_corpus_temporal_split.png", dpi=300)
plt.close()

# ===============================
# HELPER: KG NETWORK FIGURE
# ===============================

def plot_kg(edge_file, out_file, title, top_n=80):
    df = pd.read_csv(edge_file).fillna("")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

    df = df.sort_values("weight", ascending=False).head(top_n)

    G = nx.Graph()

    for _, r in df.iterrows():
        s = r["source"]
        t = r["target"]
        w = r["weight"]

        G.add_edge(s, t, weight=w)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.6)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 4 for w in weights]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=350,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.35
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_file, dpi=300)
    plt.close()

plot_kg(
    KG_DIR / "dino_pre2016_semantic_edges.csv",
    "02_dino_pre2016_kg.png",
    "Dinoflagellate pre-2016 semantic KG"
)

plot_kg(
    KG_DIR / "dino_post2015_semantic_edges.csv",
    "03_dino_post2015_kg.png",
    "Dinoflagellate post-2015 semantic KG"
)

plot_kg(
    KG_DIR / "cyano_all_semantic_edges.csv",
    "04_cyano_semantic_kg.png",
    "Cyanobacterial STX semantic KG"
)

# ===============================
# 3. CONSERVED VS DIVERGENT BAR PLOT
# ===============================

summary = pd.read_csv(
    CROSS_DIR / "cross_taxa_transfer_summary.csv"
).fillna("")

needed = [
    "cyano_all_vs_dino_all_conserved_edges",
    "dino_all_edges",
    "cyano_all_edges"
]

summary_dict = dict(zip(summary["analysis"], summary["count"]))

conserved = int(summary_dict.get("cyano_all_vs_dino_all_conserved_edges", 0))
dino_all = int(summary_dict.get("dino_all_edges", 0))
cyano_all = int(summary_dict.get("cyano_all_edges", 0))

dino_specific = max(dino_all - conserved, 0)
cyano_specific = max(cyano_all - conserved, 0)

cvd = pd.DataFrame({
    "Category": [
        "Conserved",
        "Dino-specific",
        "Cyano-specific"
    ],
    "Edges": [
        conserved,
        dino_specific,
        cyano_specific
    ]
})

plt.figure(figsize=(7, 5))
plt.bar(cvd["Category"], cvd["Edges"])
plt.ylabel("Number of semantic edges")
plt.title("Conserved vs divergent cross-taxa STX semantic edges")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_conserved_vs_divergent_edges.png", dpi=300)
plt.close()

# ===============================
# 4. TRANSFER CATEGORY BAR PLOT
# ===============================

transfer = pd.read_csv(
    CROSS_DIR / "cyano_plus_dino_pre2016_predicts_dino_post2015.csv"
).fillna("")

transfer_counts = transfer["transfer_type"].value_counts().reset_index()
transfer_counts.columns = ["Transfer category", "Count"]

plt.figure(figsize=(8, 5))
plt.bar(
    transfer_counts["Transfer category"],
    transfer_counts["Count"]
)
plt.ylabel("Number of post-2015 dino edges")
plt.title("Cross-taxa transfer categories")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_cross_taxa_transfer_categories.png", dpi=300)
plt.close()

# ===============================
# 5. NODE2VEC VS AI COMPARISON
# ===============================

compare = pd.read_csv(
    ML_DIR / "node2vec_vs_ai_comparison_metrics.csv"
).fillna("")

plt.figure(figsize=(7, 5))

for method in compare["Method"].unique():
    sub = compare[compare["Method"] == method]
    plt.plot(
        sub["K"],
        sub["Precision@K"],
        marker="o",
        label=method
    )

plt.xlabel("Top K")
plt.ylabel("Precision@K")
plt.title("Node2Vec vs supervised AI ranking")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "07_node2vec_vs_ai_precision.png", dpi=300)
plt.close()

# ===============================
# 6. STRICT TEMPORAL VALIDATION P@K
# ===============================

strict = pd.read_csv(
    ML_DIR / "strict_temporal_validation_metrics.csv"
).fillna("")

plt.figure(figsize=(7, 5))
plt.plot(
    strict["K"],
    strict["Precision@K"],
    marker="o"
)
plt.xlabel("Top K")
plt.ylabel("Precision@K")
plt.title("Strict temporal validation of Node2Vec-ranked hypotheses")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_strict_temporal_precision_at_k.png", dpi=300)
plt.close()

# ===============================
# 7. SAVE FIGURE INDEX
# ===============================

figure_index = pd.DataFrame({
    "figure": [
        "01_corpus_temporal_split.png",
        "02_dino_pre2016_kg.png",
        "03_dino_post2015_kg.png",
        "04_cyano_semantic_kg.png",
        "05_conserved_vs_divergent_edges.png",
        "06_cross_taxa_transfer_categories.png",
        "07_node2vec_vs_ai_precision.png",
        "08_strict_temporal_precision_at_k.png"
    ],
    "description": [
        "Corpus distribution across dinoflagellate pre-2016, dinoflagellate post-2015, and cyanobacterial corpus.",
        "Top weighted dinoflagellate pre-2016 semantic KG.",
        "Top weighted dinoflagellate post-2015 semantic KG.",
        "Top weighted cyanobacterial STX semantic KG.",
        "Conserved and divergent semantic edges between cyanobacteria and dinoflagellates.",
        "Categories of post-2015 dinoflagellate edges supported by cyanobacterial and/or pre-2016 dinoflagellate prior knowledge.",
        "Precision@K comparison between Node2Vec ranking and supervised AI ranking.",
        "Strict temporal validation Precision@K for Node2Vec-ranked pre-2016 hypotheses."
    ]
})

figure_index.to_csv(
    FIG_DIR / "figure_index.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== FINAL FIGURES GENERATED ==========")
print("Saved figures to:")
print(FIG_DIR)

print("\nFigures:")
print(figure_index.to_string(index=False))
