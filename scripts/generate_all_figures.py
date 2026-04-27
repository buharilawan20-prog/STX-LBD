import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
DPI = 300
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# PATHS
# You are running from: ~/LBD/New/Cyano
# Dino project is one folder back: ~/LBD/New/New
# =========================
DINO_DIR = Path("../New")

PATH_CORPUS = DINO_DIR / "data/processed/dino_all_clean.csv"
PATH_GRAPH = DINO_DIR / "data/graphs_semantic/semantic_pre2016_edges_filtered.csv"
PATH_TEMPORAL = DINO_DIR / "results/temporal_validation/temporal_validation_metrics.csv"

PATH_TRANSFER_SUMMARY = Path("results/cyano_transfer/biological_insights/category_summary.csv")
PATH_TRANSFER_BASIC = Path("results/cyano_transfer/summary.csv")


def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.jpg", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=DPI, bbox_inches="tight")


def check_file(path):
    if not Path(path).exists():
        print(f"Missing file: {path}")
        return False
    return True


# =========================
# FIGURE 2 — CORPUS YEAR DISTRIBUTION
# =========================
def fig_corpus():
    if not check_file(PATH_CORPUS):
        return

    df = pd.read_csv(PATH_CORPUS, encoding="utf-8-sig")

    fig = plt.figure(figsize=(8, 5))
    df["year"].dropna().astype(int).hist(bins=30)

    plt.axvline(x=2015, linestyle="--", linewidth=1.5)
    plt.xlabel("Publication year")
    plt.ylabel("Number of papers")
    plt.title("Dinoflagellate STX corpus distribution with 2015 cutoff")

    save_figure(fig, "Figure2_corpus_distribution")
    plt.close()


# =========================
# FIGURE 4 — TOP CENTRAL NODES
# =========================
def fig_top_nodes():
    if not check_file(PATH_GRAPH):
        return

    df = pd.read_csv(PATH_GRAPH, encoding="utf-8-sig", low_memory=False)

    deg = {}

    for _, row in df.iterrows():
        w = float(row.get("Weight", 1))
        deg[row["Source"]] = deg.get(row["Source"], 0) + w
        deg[row["Target"]] = deg.get(row["Target"], 0) + w

    top = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]

    nodes = [x[0] for x in top]
    values = [x[1] for x in top]

    fig = plt.figure(figsize=(9, 7))
    plt.barh(nodes, values)
    plt.gca().invert_yaxis()

    plt.xlabel("Weighted degree")
    plt.title("Top central nodes in the pre-2016 semantic STX knowledge graph")

    save_figure(fig, "Figure4_top_central_nodes")
    plt.close()


# =========================
# FIGURE 5 — TEMPORAL VALIDATION
# =========================
def fig_temporal():
    if not check_file(PATH_TEMPORAL):
        return

    df = pd.read_csv(PATH_TEMPORAL, encoding="utf-8-sig")

    k_values = [10, 20, 50, 100, 200]

    precision = []
    hits = []

    for k in k_values:
        p = df.loc[df["Metric"] == f"Precision@{k}", "Value"]
        h = df.loc[df["Metric"] == f"Hits@{k}", "Value"]

        precision.append(float(p.iloc[0]) if len(p) else 0)
        hits.append(float(h.iloc[0]) if len(h) else 0)

    fig = plt.figure(figsize=(8, 5))

    plt.plot(k_values, precision, marker="o", label="Precision@K")
    plt.plot(k_values, hits, marker="s", label="Hits@K")

    plt.xlabel("Top-K ranked hypotheses")
    plt.ylabel("Score / count")
    plt.title("Temporal validation performance")
    plt.legend()

    save_figure(fig, "Figure5_temporal_validation")
    plt.close()


# =========================
# FIGURE 6 — CROSS-TAXA TRANSFER
# =========================
def fig_transfer():
    if check_file(PATH_TRANSFER_BASIC):
        df = pd.read_csv(PATH_TRANSFER_BASIC, encoding="utf-8-sig")
        labels = df["Analysis"].astype(str).tolist()
        values = df["Rate"].astype(float).tolist()
    else:
        labels = ["Cyano → Post-2015 Dino", "Cyano → All Dino"]
        values = [0.035951, 0.042993]

    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, values)

    plt.ylabel("Transfer / similarity rate")
    plt.title("Cross-taxa saxitoxin knowledge transfer")
    plt.xticks(rotation=20, ha="right")

    save_figure(fig, "Figure6_cross_taxa_transfer")
    plt.close()


# =========================
# FIGURE 7 — CONSERVED VS DIVERGENT BIOLOGY
# =========================
def fig_conserved_divergent():
    if not check_file(PATH_TRANSFER_SUMMARY):
        return

    df = pd.read_csv(PATH_TRANSFER_SUMMARY, encoding="utf-8-sig")

    if "Transfer_Match" not in df.columns or "Category" not in df.columns:
        print("category_summary.csv does not contain expected columns: Transfer_Match, Category")
        return

    pivot = df.pivot(index="Category", columns="Transfer_Match", values="Count").fillna(0)

    # Rename columns safely
    rename_map = {}
    for col in pivot.columns:
        if int(col) == 0:
            rename_map[col] = "Divergent / cyano-only"
        elif int(col) == 1:
            rename_map[col] = "Conserved / transferred"

    pivot = pivot.rename(columns=rename_map)

    fig = plt.figure(figsize=(9, 6))
    pivot.plot(kind="bar", ax=plt.gca())

    plt.ylabel("Number of relationships")
    plt.xlabel("Biological category")
    plt.title("Conserved versus divergent STX biology")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Relationship class")

    save_figure(fig, "Figure7_conserved_vs_divergent")
    plt.close()


# =========================
# RUN ALL
# =========================
def main():
    print("Generating all figures...")

    fig_corpus()
    fig_top_nodes()
    fig_temporal()
    fig_transfer()
    fig_conserved_divergent()

    print(f"All available figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
