from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# =========================
# FILE PATHS
# =========================
CORPUS = "data/processed/dino_all_clean.csv"
NODES = "data/graphs/dino_pre2016_nodes.csv"
EDGES = "data/graphs/dino_pre2016_edges.csv"
METRICS = "results/temporal_validation/temporal_validation_metrics.csv"
VALIDATION = "results/temporal_validation/temporal_validation_results.csv"

OUTDIR = "results/publication_figures"


# =========================
# FIGURE 1 — WORKFLOW
# =========================
def figure_workflow(outdir):
    plt.figure(figsize=(10, 2))

    steps = [
        "Literature",
        "Entity\nExtraction",
        "Knowledge\nGraph",
        "Node2Vec",
        "Hypothesis\nRanking",
        "Temporal\nValidation"
    ]

    for i, step in enumerate(steps):
        plt.text(i, 0.5, step, ha='center', va='center', fontsize=12)
        if i < len(steps) - 1:
            plt.arrow(i + 0.3, 0.5, 0.4, 0, head_width=0.05)

    plt.axis('off')
    plt.title("Workflow of STX-AI Pipeline")
    plt.savefig(outdir / "Figure1_Workflow.png", dpi=300)
    plt.close()


# =========================
# FIGURE 2 — YEAR DIST
# =========================
def figure_year_distribution(df, outdir):
    plt.figure()

    counts = df["year"].value_counts().sort_index()
    counts.plot(kind="bar")

    if 2015 in counts.index:
        cutoff_index = list(counts.index).index(2015)
        plt.axvline(x=cutoff_index, linestyle="--")

    plt.title("Corpus Distribution with 2015 Cutoff")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")

    plt.tight_layout()
    plt.savefig(outdir / "Figure2_YearDistribution.png", dpi=300)
    plt.close()


# =========================
# FIGURE 3 — GRAPH
# =========================
def figure_graph(nodes, edges, outdir):
    G = nx.Graph()

    for _, row in edges.iterrows():
        G.add_node(row["Source"], entity_type=row["Source_Type"])
        G.add_node(row["Target"], entity_type=row["Target_Type"])
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

    color_map = {
        "TOXIN": "red",
        "GENE": "blue",
        "SPECIES": "green",
        "ENV_FACTOR": "orange",
        "PROCESS": "purple"
    }

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 10))

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[color_map.get(G.nodes[n]["entity_type"], "gray") for n in G.nodes()],
        node_size=200
    )

    plt.title("Pre-2016 STX Knowledge Graph")
    plt.axis("off")
    plt.savefig(outdir / "Figure3_KnowledgeGraph.png", dpi=300)
    plt.close()


# =========================
# FIGURE 4 — TOP NODES
# =========================
def figure_top_nodes(nodes, outdir):
    top = nodes.sort_values("Weighted_Degree", ascending=False).head(15)

    plt.figure()
    plt.barh(top["Node"], top["Weighted_Degree"])
    plt.gca().invert_yaxis()

    plt.title("Top Central Biological Nodes")
    plt.xlabel("Weighted Degree")

    plt.tight_layout()
    plt.savefig(outdir / "Figure4_TopNodes.png", dpi=300)
    plt.close()


# =========================
# FIGURE 5 — PERFORMANCE
# =========================
def figure_performance(metrics, outdir):
    precision = metrics[metrics["Metric"].str.contains("Precision@")]
    hits = metrics[metrics["Metric"].str.contains("Hits@")]

    plt.figure()

    plt.plot(precision["Metric"], precision["Value"], marker='o', label="Precision")
    plt.plot(hits["Metric"], hits["Value"], marker='o', label="Hits")

    plt.xticks(rotation=45)
    plt.title("Temporal Validation Performance")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "Figure5_Performance.png", dpi=300)
    plt.close()


# =========================
# FIGURE 6 — TOP VALIDATED
# =========================
def figure_top_validated(df, outdir):
    validated = df[df["Appears_in_Future"] == 1].head(15)

    labels = validated["Source"] + " — " + validated["Target"]

    plt.figure()
    plt.barh(labels, validated["Embedding_Integrated_Score"])
    plt.gca().invert_yaxis()

    plt.title("Top Validated Hypotheses")
    plt.xlabel("AI Score")

    plt.tight_layout()
    plt.savefig(outdir / "Figure6_Validated.png", dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(CORPUS)
    nodes = pd.read_csv(NODES)
    edges = pd.read_csv(EDGES)
    metrics = pd.read_csv(METRICS)
    validation = pd.read_csv(VALIDATION)

    figure_workflow(outdir)
    figure_year_distribution(corpus, outdir)
    figure_graph(nodes, edges, outdir)
    figure_top_nodes(nodes, outdir)
    figure_performance(metrics, outdir)
    figure_top_validated(validation, outdir)

    print("All publication figures saved in:", outdir)


if __name__ == "__main__":
    main()
