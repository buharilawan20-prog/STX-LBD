from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


FIGDIR = Path("figures_redo")
FIGDIR.mkdir(exist_ok=True)

DPI = 600


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGDIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGDIR / f"{name}.jpg", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGDIR / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()


# =====================================================
# FIGURE 1 — Temporal validation
# =====================================================
def fig_temporal_validation():
    df = pd.read_csv(
        "results/dino_temporal_normalized_ml/normalized_node2vec_ml_temporal_metrics.csv"
    )

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(df["K"], df["Precision@K"], marker="o", linewidth=2)
    ax1.set_xlabel("Top-K ranked hypotheses")
    ax1.set_ylabel("Precision@K")
    ax1.set_xscale("log")
    ax1.set_ylim(0, max(df["Precision@K"]) + 0.1)

    ax2 = ax1.twinx()
    ax2.plot(df["K"], df["Hits@K"], marker="s", linewidth=2, linestyle="--")
    ax2.set_ylabel("Hits@K")

    ax1.set_title("Temporal validation of AI-ranked STX hypotheses", fontweight="bold")

    savefig("Figure1_Temporal_Validation")


# =====================================================
# FIGURE 2 — ML model performance
# =====================================================
def fig_ml_performance():
    df = pd.read_csv(
        "results/dino_temporal_normalized_ml/semantic_ml_scoring_performance.csv"
    )

    x = range(len(df))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar([i - 0.18 for i in x], df["ROC_AUC"], width=0.35, label="ROC-AUC")
    ax.bar([i + 0.18 for i in x], df["PR_AUC"], width=0.35, label="PR-AUC")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Semantic ML scoring performance", fontweight="bold")
    ax.legend(frameon=False)

    savefig("Figure2_ML_Model_Performance")


# =====================================================
# FIGURE 3 — Cross taxa transfer rates
# =====================================================
def fig_cross_taxa_transfer():
    df = pd.read_csv(
        "results/cross_taxa_transfer/overall_cross_taxa_summary.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        "Cyano →\nAll Dino",
        "Cyano →\nPost-2015 Dino",
        "Cyano + Pre-2016 Dino →\nPost-2015 Dino"
    ]

    ax.bar(labels, df["Transfer_Rate"])

    for i, v in enumerate(df["Transfer_Rate"]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Transfer rate")
    ax.set_title("Cross-taxa transfer of STX relationships", fontweight="bold")

    savefig("Figure3_Cross_Taxa_Transfer_Rates")


# =====================================================
# FIGURE 4 — Top validated hypotheses network
# =====================================================
def fig_top_validated_network():
    df = pd.read_csv(
        "results/dino_temporal_normalized_ml/normalized_dino_hypotheses_node2vec_ml_ranked.csv"
    )

    df = df[df["Match"] == 1].head(15)

    G = nx.Graph()

    for _, r in df.iterrows():
        G.add_edge(
            r["Source"],
            r["Target"],
            weight=r["Final_AI_Score"]
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    pos = nx.spring_layout(G, seed=42, k=0.8)

    node_sizes = [800 + 200 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_edges(
        G, pos,
        width=1.5,
        alpha=0.6,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    ax.set_title(
        "Top validated AI-ranked STX hypotheses",
        fontweight="bold"
    )
    ax.axis("off")

    savefig("Figure4_Top_Validated_Hypothesis_Network")


# =====================================================
# Sankey helper
# =====================================================
def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.55):
    verts = [
        (x0, y0a),
        ((x0+x1)/2, y0a),
        ((x0+x1)/2, y1a),
        (x1, y1a),
        (x1, y1b),
        ((x0+x1)/2, y1b),
        ((x0+x1)/2, y0b),
        (x0, y0b),
        (x0, y0a)
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY
    ]

    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha
        )
    )


def add_node(ax, x, y, w, h, label, color, fontsize=8):
    ax.add_patch(
        Rectangle(
            (x, y), w, h,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold"
    )


# =====================================================
# FIGURE 5 — Conserved vs divergent Sankey
# =====================================================
def fig_sankey_conserved_divergent():
    conserved = pd.read_csv(
        "results/cross_taxa_transfer/cyano_to_all_dino_conserved.csv"
    ).head(6)

    divergent = pd.read_csv(
        "results/cross_taxa_transfer/cyano_to_all_dino_divergent.csv"
    ).head(6)

    conserved["Outcome"] = "Conserved"
    divergent["Outcome"] = "Divergent"

    df = pd.concat([conserved, divergent]).reset_index(drop=True)

    def category(row):
        text = f"{row['Source']} {row['Target']} {row.get('Relation','')}".lower()

        if any(x in text for x in ["temperature", "nitrogen", "phosphorus", "salinity", "ph", "light", "climate"]):
            return "Environmental"
        if any(x in text for x in ["sxta", "sxtg", "sxt genes", "gene"]):
            return "Gene-related"
        if any(x in text for x in ["biosynthesis", "pathway"]):
            return "Biosynthesis"
        if any(x in text for x in ["toxin", "saxitoxin", "pst"]):
            return "Toxin phenotype"
        return "Other"

    df["Category"] = df.apply(category, axis=1)
    df["Relationship"] = df["Source"].astype(str) + " ↔ " + df["Target"].astype(str)

    category_colors = {
        "Environmental": "#E76F51",
        "Gene-related": "#2A9D8F",
        "Biosynthesis": "#6A4C93",
        "Toxin phenotype": "#457B9D",
        "Other": "#999999",
    }

    outcome_colors = {
        "Conserved": "#2A9D8F",
        "Divergent": "#E76F51",
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x_cat, x_rel, x_out = 0.05, 0.40, 0.80
    w_cat, w_rel, w_out = 0.18, 0.24, 0.14

    y_top, y_bottom = 0.85, 0.12
    gap = 0.015
    n = len(df)
    h = (y_top - y_bottom - gap * (n - 1)) / n

    rel_pos = {}
    for i, r in df.iterrows():
        y0 = y_top - i * (h + gap) - h
        y1 = y0 + h
        rel_pos[i] = (y0, y1)

        label = r["Relationship"]
        if len(label) > 28:
            label = label[:25] + "..."

        add_node(
            ax, x_rel, y0, w_rel, h,
            f"{label}\nW={int(r['Weight'])}",
            "white",
            fontsize=7.5
        )

    cat_counts = df.groupby("Category")["Weight"].sum()
    cat_pos = {}
    y = y_top
    for cat, wt in cat_counts.items():
        hh = (wt / cat_counts.sum()) * (y_top - y_bottom)
        y0, y1 = y - hh, y
        cat_pos[cat] = {"y0": y0, "y1": y1, "current": y1}
        add_node(
            ax, x_cat, y0, w_cat, hh,
            cat,
            category_colors.get(cat, "#999999"),
            fontsize=8
        )
        y = y0 - 0.025

    out_counts = df.groupby("Outcome")["Weight"].sum()
    out_pos = {}
    y = y_top
    for out in ["Conserved", "Divergent"]:
        if out not in out_counts:
            continue
        hh = (out_counts[out] / out_counts.sum()) * (y_top - y_bottom - 0.08)
        y0, y1 = y - hh, y
        out_pos[out] = {"y0": y0, "y1": y1, "current": y1}
        add_node(
            ax, x_out, y0, w_out, hh,
            f"{out}\n(n={sum(df['Outcome']==out)})",
            outcome_colors[out],
            fontsize=9
        )
        y = y0 - 0.08

    for i, r in df.iterrows():
        cat = r["Category"]
        wt = r["Weight"]
        ry0, ry1 = rel_pos[i]

        hh = (cat_pos[cat]["y1"] - cat_pos[cat]["y0"]) * wt / cat_counts[cat]
        cy1 = cat_pos[cat]["current"]
        cy0 = cy1 - hh
        cat_pos[cat]["current"] = cy0

        ribbon(
            ax,
            x_cat + w_cat, cy0, cy1,
            x_rel, ry0, ry1,
            category_colors.get(cat, "#999999"),
            alpha=0.35
        )

    for i, r in df.iterrows():
        out = r["Outcome"]
        wt = r["Weight"]
        ry0, ry1 = rel_pos[i]

        hh = (out_pos[out]["y1"] - out_pos[out]["y0"]) * wt / out_counts[out]
        oy1 = out_pos[out]["current"]
        oy0 = oy1 - hh
        out_pos[out]["current"] = oy0

        ribbon(
            ax,
            x_rel + w_rel, ry0, ry1,
            x_out, oy0, oy1,
            outcome_colors[out],
            alpha=0.55
        )

    ax.text(
        0.05, 0.96,
        "Cross-taxa conservation and divergence of STX relationships",
        fontsize=15,
        fontweight="bold"
    )

    ax.text(0.05, 0.89, "Biological category", fontsize=11, fontweight="bold")
    ax.text(0.40, 0.89, "Representative relationship", fontsize=11, fontweight="bold")
    ax.text(0.80, 0.89, "Outcome", fontsize=11, fontweight="bold")

    savefig("Figure5_Conserved_Divergent_Sankey")


# =====================================================
# FIGURE 6 — Normalized KG comparison
# =====================================================
def fig_kg_comparison():
    files = {
        "Cyanobacteria": "data/graphs_cyano/cyano_semantic_edges_normalized_collapsed.csv",
        "Dinoflagellates": "data/graphs_dino/dino_semantic_edges_normalized_collapsed.csv",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (title, file) in zip(axes, files.items()):
        df = pd.read_csv(file).head(120)

        G = nx.Graph()

        for _, r in df.iterrows():
            G.add_edge(r["Source"], r["Target"], weight=r["Weight"])

        pos = nx.spring_layout(G, seed=42, k=0.4)

        sizes = [60 + G.degree(n) * 25 for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=sizes, ax=ax)

        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:15]
        labels = {n: n for n, d in top_nodes}

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    savefig("Figure6_Normalized_KG_Comparison")


def main():
    print("Generating redo figures...")

    fig_temporal_validation()
    fig_ml_performance()
    fig_cross_taxa_transfer()
    fig_top_validated_network()
    fig_sankey_conserved_divergent()
    fig_kg_comparison()

    print("Done. Figures saved in figures_redo/")


if __name__ == "__main__":
    main()
