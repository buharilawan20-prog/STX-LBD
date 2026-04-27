from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


OUTDIR = Path("results/final_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

ENTITIES = "data/processed/entities_semantic_merged.csv"
EDGES = "data/graphs_semantic/semantic_pre2016_edges_filtered.csv"
SHORTLIST_SUMMARY = "results/semantic_hypotheses/STX_DISCOVERY_SHORTLIST_SUMMARY.csv"
VALIDATION_SUMMARY = "results/discovery_validation/discovery_validation_summary.csv"
ML_METRICS = "results/final_ai_discoveries/semantic_ml_scoring_metrics.csv"
FINAL_DISCOVERIES = "results/final_ai_discoveries/FINAL_AI_SCORED_DISCOVERY_SHORTLIST.csv"


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUTDIR / name.replace(".png", ".pdf"), format="pdf")
    plt.close()


def fig1_workflow():
    steps = [
        "Literature\nCorpus",
        "Semantic Entity\nExtraction",
        "N-gram\nPhrase Mining",
        "Semantic\nKnowledge Graph",
        "Hypothesis\nGeneration",
        "ML\nScoring",
        "Temporal\nValidation",
        "Discovery\nShortlist",
    ]

    plt.figure(figsize=(16, 3))
    for i, step in enumerate(steps):
        plt.text(
            i, 0.5, step,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
        )
        if i < len(steps) - 1:
            plt.arrow(i + 0.32, 0.5, 0.32, 0, head_width=0.05, length_includes_head=True)

    plt.axis("off")
    plt.title("STX-AI Semantic Knowledge Graph Discovery Workflow")
    savefig("Figure1_Workflow.png")


def fig2_entity_composition():
    df = pd.read_csv(ENTITIES, encoding="utf-8-sig")
    counts = df["entity_type"].value_counts().head(20).sort_values()

    plt.figure(figsize=(9, 7))
    counts.plot(kind="barh")
    plt.xlabel("Entity mentions")
    plt.ylabel("Entity type")
    plt.title("Semantic Entity Composition")
    savefig("Figure2_SemanticEntityComposition.png")


def fig3_semantic_graph():
    edges = pd.read_csv(EDGES, encoding="utf-8-sig")

    # Keep strongest edges only for readable network
    edges = edges.sort_values("Weight", ascending=False).head(250)

    G = nx.Graph()

    for _, row in edges.iterrows():
        G.add_node(row["Source"], entity_type=row["Source_Type"])
        G.add_node(row["Target"], entity_type=row["Target_Type"])
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

    color_map = {
        "GENE": "blue",
        "GENE_DOMAIN": "deepskyblue",
        "GENE_PRESENCE_ABSENCE": "navy",
        "REGULATION_EXPRESSION": "purple",
        "EVOLUTIONARY_PROCESS": "darkgreen",
        "BIOLOGICAL_MECHANISM": "orange",
        "BIOSYNTHETIC_SYSTEM": "gold",
        "ENV_FACTOR": "red",
        "SPECIES": "green",
        "TOXIN": "gray",
        "TOXIN_PHENOTYPE": "brown",
        "PHRASE_GENE_MECHANISM": "lightblue",
        "PHRASE_TOXIN_PHENOTYPE": "pink",
        "PHRASE_ENVIRONMENT": "salmon",
    }

    weighted_degree = dict(G.degree(weight="weight"))
    node_sizes = [80 + weighted_degree.get(n, 1) * 6 for n in G.nodes()]
    node_colors = [color_map.get(G.nodes[n].get("entity_type", ""), "lightgray") for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=0.55)

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.18, width=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)

    # label top nodes only
    top_nodes = sorted(weighted_degree, key=weighted_degree.get, reverse=True)[:25]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Filtered Pre-2016 Semantic STX Knowledge Graph")
    plt.axis("off")
    savefig("Figure3_FilteredSemanticKnowledgeGraph.png")


def fig4_discovery_categories():
    df = pd.read_csv(SHORTLIST_SUMMARY, encoding="utf-8-sig")
    df = df.sort_values("Count")

    plt.figure(figsize=(9, 6))
    plt.barh(df["Biological_Category"], df["Count"])
    plt.xlabel("Number of shortlisted hypotheses")
    plt.ylabel("Biological category")
    plt.title("Discovery Shortlist by Biological Category")
    savefig("Figure4_DiscoveryCategories.png")


def fig5_validation_rate():
    df = pd.read_csv(VALIDATION_SUMMARY, encoding="utf-8-sig")
    df = df.sort_values("Validation_Rate")

    plt.figure(figsize=(9, 6))
    plt.barh(df["Biological_Category"], df["Validation_Rate"])
    plt.xlabel("Validation rate in post-2015 literature")
    plt.ylabel("Biological category")
    plt.title("Temporal Validation of Discovery Hypotheses")
    savefig("Figure5_DiscoveryValidationRate.png")


def fig6_ml_performance():
    df = pd.read_csv(ML_METRICS, encoding="utf-8-sig")

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], df["ROC_AUC"], width, label="ROC-AUC")
    plt.bar([i + width/2 for i in x], df["PR_AUC"], width, label="PR-AUC")

    plt.xticks(x, df["Model"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("ML Scoring Performance for Semantic Hypotheses")
    plt.legend()
    savefig("Figure6_MLPerformance.png")


def fig7_top_discoveries():
    df = pd.read_csv(FINAL_DISCOVERIES, encoding="utf-8-sig")
    top = df.sort_values("Final_AI_Discovery_Score", ascending=False).head(15).copy()

    labels = top["Source"].astype(str) + " ↔ " + top["Target"].astype(str)

    plt.figure(figsize=(10, 8))
    plt.barh(labels[::-1], top["Final_AI_Discovery_Score"][::-1])
    plt.xlabel("Final AI discovery score")
    plt.title("Top AI-ranked STX Discovery Hypotheses")
    savefig("Figure7_TopFinalDiscoveries.png")


def main():
    fig1_workflow()
    fig2_entity_composition()
    fig3_semantic_graph()
    fig4_discovery_categories()
    fig5_validation_rate()
    fig6_ml_performance()
    fig7_top_discoveries()

    print(f"Final figures saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
