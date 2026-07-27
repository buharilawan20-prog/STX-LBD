import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG_DIR = BASE / "FINAL_WORKSPACE/kg"
FIG_DIR = BASE / "FINAL_WORKSPACE/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DINO_EDGE_FILE = KG_DIR / "dino_pre2016_semantic_edges.csv"
CYANO_EDGE_FILE = KG_DIR / "cyano_all_semantic_edges.csv"

OUT_PNG = FIG_DIR / "Fig2_dino_cyano_KG_style.png"
OUT_PDF = FIG_DIR / "Fig2_dino_cyano_KG_style.pdf"

TYPE_COLORS = {
    "TOXIN": "#D55E00",
    "SXT_GENE": "#0072B2",
    "DINO_TAXON": "#009E73",
    "CYANO_TAXON": "#56B4E9",
    "ENV_FACTOR": "#E69F00",
    "BIOLOGICAL_PROCESS": "#CC79A7",
    "DETECTION_METHOD": "#999999",
    "OTHER": "#BBBBBB"
}

def build_graph(edge_file, top_n=120):
    df = pd.read_csv(edge_file).fillna("")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

    df = df.sort_values("weight", ascending=False).head(top_n)

    G = nx.Graph()

    for _, r in df.iterrows():
        s = str(r["source"]).strip()
        t = str(r["target"]).strip()

        if not s or not t or s == t:
            continue

        s_type = str(r.get("source_type", "OTHER")).strip()
        t_type = str(r.get("target_type", "OTHER")).strip()

        w = float(r["weight"])

        G.add_node(s, node_type=s_type)
        G.add_node(t, node_type=t_type)

        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    return G

def draw_panel(ax, G, title, panel_label):
    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.55,
        iterations=200,
        weight="weight"
    )

    degrees = dict(G.degree(weight="weight"))

    node_sizes = [
        80 + (degrees[n] ** 0.65) * 28
        for n in G.nodes()
    ]

    node_colors = [
        TYPE_COLORS.get(G.nodes[n].get("node_type", "OTHER"), TYPE_COLORS["OTHER"])
        for n in G.nodes()
    ]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    edge_widths = [
        0.2 + (w / max_w) * 2.2
        for w in edge_weights
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.22,
        edge_color="gray"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.4,
        edgecolors="white"
    )

    # Label only top hub nodes
    top_labels = sorted(
        degrees,
        key=degrees.get,
        reverse=True
    )[:22]

    labels = {n: n for n in top_labels}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_family="DejaVu Sans"
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    ax.text(
        -0.03,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.axis("off")

# ===============================
# BUILD GRAPHS
# ===============================

G_dino = build_graph(DINO_EDGE_FILE, top_n=130)
G_cyano = build_graph(CYANO_EDGE_FILE, top_n=130)

# ===============================
# DRAW FIGURE
# ===============================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 15)
)

draw_panel(
    axes[0],
    G_dino,
    "Dinoflagellate pre-2016 STX semantic knowledge graph",
    "A"
)

draw_panel(
    axes[1],
    G_cyano,
    "Cyanobacterial STX semantic knowledge graph",
    "B"
)

# ===============================
# LEGEND
# ===============================

legend_items = []

for entity_type, color in TYPE_COLORS.items():
    if entity_type == "OTHER":
        continue

    legend_items.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=entity_type.replace("_", " ").title(),
            markerfacecolor=color,
            markersize=9
        )
    )

fig.legend(
    handles=legend_items,
    loc="center right",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    title="Entity type",
    fontsize=9,
    title_fontsize=10
)

fig.suptitle(
    "Cross-taxa STX semantic knowledge graph structure",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(rect=[0, 0, 0.86, 0.98])

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nDino KG:")
print("Nodes:", G_dino.number_of_nodes())
print("Edges:", G_dino.number_of_edges())

print("\nCyano KG:")
print("Nodes:", G_cyano.number_of_nodes())
print("Edges:", G_cyano.number_of_edges())
