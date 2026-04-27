from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# INPUT FILES
CORPUS_FILE = "data/processed/dino_all_clean.csv"
ENTITY_FILE = "data/processed/entities_normalized.csv"
NODE_FILE = "data/graphs/dino_pre2016_nodes.csv"
METRICS_FILE = "results/temporal_validation/temporal_validation_metrics.csv"

# OUTPUT DIR
OUTPUT_DIR = "results/figures"


def plot_year_distribution(df, outdir):
    plt.figure()
    df["year"].value_counts().sort_index().plot(kind="bar")
    plt.axvline(x=list(df["year"].sort_values().unique()).index(2015), linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Corpus Distribution by Year")
    plt.tight_layout()
    plt.savefig(outdir / "figure_year_distribution.png")
    plt.close()


def plot_entity_types(df, outdir):
    plt.figure()
    df["entity_type"].value_counts().plot(kind="bar")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.title("Entity Type Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "figure_entity_types.png")
    plt.close()


def plot_top_entities(df, outdir):
    top = df["entity_normalized"].value_counts().head(20)

    plt.figure()
    top.sort_values().plot(kind="barh")
    plt.xlabel("Frequency")
    plt.title("Top 20 Entities")
    plt.tight_layout()
    plt.savefig(outdir / "figure_top_entities.png")
    plt.close()


def plot_top_nodes(df, outdir):
    top = df.sort_values("Weighted_Degree", ascending=False).head(20)

    plt.figure()
    plt.barh(top["Node"], top["Weighted_Degree"])
    plt.xlabel("Weighted Degree")
    plt.title("Top 20 Central Nodes (Pre-2016 Graph)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "figure_top_nodes.png")
    plt.close()


def plot_temporal_metrics(df, outdir):
    precision = df[df["Metric"].str.contains("Precision@")]

    plt.figure()
    plt.plot(
        precision["Metric"],
        precision["Value"],
        marker="o"
    )
    plt.xlabel("Metric")
    plt.ylabel("Precision")
    plt.title("Temporal Validation Performance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outdir / "figure_temporal_performance.png")
    plt.close()


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus_df = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig")
    entity_df = pd.read_csv(ENTITY_FILE, encoding="utf-8-sig")
    node_df = pd.read_csv(NODE_FILE, encoding="utf-8-sig")
    metrics_df = pd.read_csv(METRICS_FILE, encoding="utf-8-sig")

    plot_year_distribution(corpus_df, outdir)
    plot_entity_types(entity_df, outdir)
    plot_top_entities(entity_df, outdir)
    plot_top_nodes(node_df, outdir)
    plot_temporal_metrics(metrics_df, outdir)

    print("Figures saved in:", outdir)


if __name__ == "__main__":
    main()
