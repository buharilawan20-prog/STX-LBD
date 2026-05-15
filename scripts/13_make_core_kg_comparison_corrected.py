from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTDIR = Path("figures_redo")
OUTDIR.mkdir(exist_ok=True)

DINO_FILE = "data/graphs_dino/dino_semantic_edges_normalized_collapsed.csv"
CYANO_FILE = "data/graphs_cyano/cyano_semantic_edges_normalized_collapsed.csv"

DPI = 600

TOP_EDGES = 140
MIN_WEIGHT = 4
MAX_LABELS = 18

COLORS = {
    "GENE": "#1f77b4",
    "TOXIN": "#6e6e6e",
    "ENV_FACTOR": "#d62728",
    "BIOLOGICAL_MECHANISM": "#2ca02c",
    "BIOSYNTHETIC_SYSTEM": "#ff7f0e",
    "EVOLUTIONARY_PROCESS": "#9467bd",
    "REGULATION_EXPRESSION": "#8c564b",
    "SPECIES": "#31a354",
    "TOXIN_PHENOTYPE": "#e377c2",
    "OTHER": "#bdbdbd",
}

REMOVE_TERMS = {
    "toxic",
    "non-toxic",
    "toxigenic",
    "toxic cyanobacteria",
    "toxic dinoflagellate",
    "toxin-producing",
    "producing dinoflagellates",
    "cyanobacterial",
    "dinoflagellate",
}

VISUAL_RENAME = {
    "alexandrium": "Alexandrium spp.",
    "gymnodinium": "Gymnodinium spp.",
    "pyrodinium": "Pyrodinium spp.",
    "cyanobacteria": "Cyanobacteria",
    "dinoflagellates": "Dinoflagellates",
    "sxta": "sxtA",
    "sxtg": "sxtG",
    "sxt genes": "sxt genes",
    "ph": "pH",
}


def clean_label(x):
    x = str(x).strip()
    return VISUAL_RENAME.get(x.lower(), x)


def infer_type(node):
    n = str(node).lower()

    if any(x in n for x in ["sxta", "sxtg", "sxti", "sxt genes", "sxt"]):
        return "GENE"

    if any(x in n for x in [
        "saxitoxin", "paralytic shellfish toxins",
        "neosaxitoxin", "gonyautoxins", "dcstx", "neostx"
    ]):
        return "TOXIN"

    if any(x in n for x in [
        "temperature", "nitrogen", "phosphorus", "salinity",
        "ph", "light", "climate", "oxidative"
    ]):
        return "ENV_FACTOR"

    if any(x in n for x in [
        "toxin profile", "toxin production", "toxin content",
        "toxin composition"
    ]):
        return "TOXIN_PHENOTYPE"

    if any(x in n for x in [
        "biosynthesis", "pathway", "biosynthetic"
    ]):
        return "BIOSYNTHETIC_SYSTEM"

    if any(x in n for x in [
        "evolution", "phylogeny", "gene transfer", "horizontal"
    ]):
        return "EVOLUTIONARY_PROCESS"

    if any(x in n for x in [
        "expression", "regulation"
    ]):
        return "REGULATION_EXPRESSION"

    if any(x in n for x in [
        "alexandrium", "gymnodinium", "pyrodinium",
        "cyanobacteria", "dinoflagellates"
    ]):
        return "SPECIES"

    return "OTHER"


def load_core_graph(file):
    df = pd.read_csv(file, low_memory=False)

    df = df[df["Weight"] >= MIN_WEIGHT].copy()

    df["Source"] = df["Source"].apply(clean_label)
    df["Target"] = df["Target"].apply(clean_label)

    df = df[
        ~df["Source"].str.lower().isin(REMOVE_TERMS)
        & ~df["Target"].str.lower().isin(REMOVE_TERMS)
    ].copy()

    df = df.sort_values("Weight", ascending=False).head(TOP_EDGES)

    G = nx.Graph()

    for _, r in df.iterrows():
        s = r["Source"]
        t = r["Target"]

        if s == t:
            continue

        G.add_edge(s, t, weight=float(r["Weight"]))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # keep largest connected component for clean layout
    if nx.number_connected_components(G) > 1:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    for n in G.nodes():
        G.nodes[n]["type"] = infer_type(n)

    return G


def draw_panel(ax, G, title):
    pos = nx.kamada_kawai_layout(G, weight="weight")

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    node_sizes = [
        80 + (degrees[n] / max_deg) * 900
        for n in G.nodes()
    ]

    node_colors = [
        COLORS.get(G.nodes[n]["type"], COLORS["OTHER"])
        for n in G.nodes()
    ]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1

    edge_widths = [
        0.3 + (w / max_w) * 2.5
        for w in weights
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#b0b0b0",
        width=edge_widths,
        alpha=0.35,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.95,
    )

    top_nodes = sorted(
        degrees.items(),
        key=lambda x: x[1],
        reverse=True
    )[:MAX_LABELS]

    labels = {n: n for n, d in top_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_weight="bold",
    )

    ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
    ax.axis("off")


def main():
    G_dino = load_core_graph(DINO_FILE)
    G_cyano = load_core_graph(CYANO_FILE)

    print("Dino graph:", G_dino.number_of_nodes(), "nodes,", G_dino.number_of_edges(), "edges")
    print("Cyano graph:", G_cyano.number_of_nodes(), "nodes,", G_cyano.number_of_edges(), "edges")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    draw_panel(
        axes[0],
        G_dino,
        "A. Dinoflagellate STX semantic knowledge graph"
    )

    draw_panel(
        axes[1],
        G_cyano,
        "B. Cyanobacteria STX semantic knowledge graph"
    )

    legend_items = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for label, color in COLORS.items()
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    plt.subplots_adjust(bottom=0.13, wspace=0.05)

    outbase = OUTDIR / "Figure_KG_Dino_Cyano_Comparison_Corrected"

    fig.savefig(f"{outbase}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{outbase}.jpg", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{outbase}.pdf", dpi=DPI, bbox_inches="tight")

    print("Saved:")
    print(f"{outbase}.png")
    print(f"{outbase}.jpg")
    print(f"{outbase}.pdf")


if __name__ == "__main__":
    main()
