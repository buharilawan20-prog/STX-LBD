import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG_DIR = BASE / "FINAL_WORKSPACE/kg"

FIG_DIR = BASE / "FINAL_WORKSPACE/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# INPUT FILES
# ==========================================================

# ALL dinoflagellate STX semantic KG
DINO_EDGE_FILE = KG_DIR / "dino_all_semantic_edges_taxa_normalized.csv"

# ALL cyanobacterial STX semantic KG
CYANO_EDGE_FILE = KG_DIR / "cyano_all_semantic_edges_taxa_normalized.csv"

# ==========================================================
# OUTPUT FILES
# ==========================================================

OUT_PNG = FIG_DIR / "Fig2_dino_all_vs_cyano_all_KG.png"

OUT_PDF = FIG_DIR / "Fig2_dino_all_vs_cyano_all_KG.pdf"

# ==========================================================
# ENTITY COLORS
# ==========================================================

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

# ==========================================================
# ENTITY LABELS
# ==========================================================

LABEL_MAP = {

    "TOXIN": "Toxin",

    "SXT_GENE": "sxt Gene",

    "DINO_TAXON": "Dino Taxon",

    "CYANO_TAXON": "Cyano Taxon",

    "ENV_FACTOR": "Env Factor",

    "BIOLOGICAL_PROCESS": "Biological Process",

    "DETECTION_METHOD": "Detection Method",

    "OTHER": "Other"
}

# ==========================================================
# CLEAN NODE LABELS
# ==========================================================

def clean_label(x):

    x = str(x)

    replacements = {

        "sxta": "sxtA",

        "sxtg": "sxtG",

        "sxtd": "sxtD",

        "sxti": "sxtI",

        "sxt_genes": "sxt genes",

        "paralytic_shellfish_toxins":
            "paralytic shellfish toxins",

        "paralytic_shellfish_poisoning":
            "paralytic shellfish poisoning",

        "gymnodinium_catenatum":
            "Gymnodinium catenatum",

        "alexandrium_catenella":
            "Alexandrium catenella",

        "alexandrium_tamarense":
            "Alexandrium tamarense",

        "alexandrium_minutum":
            "Alexandrium minutum",

        "alexandrium_fundyense":
            "Alexandrium fundyense",

        "pyrodinium_bahamense":
            "Pyrodinium bahamense",

        "toxin_production":
            "toxin production",

        "saxitoxin_biosynthesis":
            "STX biosynthesis",

        "toxin_biosynthesis":
            "toxin biosynthesis",

        "mass_spectrometry":
            "mass spectrometry",

        "mouse_bioassay":
            "mouse bioassay",

        "lc_ms":
            "LC-MS",

        "lc_ms_ms":
            "LC-MS/MS",

        "hplc":
            "HPLC",

        "elisa":
            "ELISA",

        "gtx":
            "GTX"
    }

    return replacements.get(
        x,
        x.replace("_", " ")
    )

# ==========================================================
# BUILD GRAPH
# ==========================================================

def build_graph(edge_file, top_n=120):

    df = pd.read_csv(edge_file).fillna("")

    df["weight"] = pd.to_numeric(
        df["weight"],
        errors="coerce"
    ).fillna(1)

    # strongest semantic edges only
    df = df.sort_values(
        "weight",
        ascending=False
    ).head(top_n)

    G = nx.Graph()

    for _, r in df.iterrows():

        s = str(r["source"]).strip()

        t = str(r["target"]).strip()

        if not s or not t or s == t:
            continue

        s_type = str(
            r.get("source_type", "OTHER")
        ).strip()

        t_type = str(
            r.get("target_type", "OTHER")
        ).strip()

        w = float(r["weight"])

        G.add_node(
            s,
            node_type=s_type
        )

        G.add_node(
            t,
            node_type=t_type
        )

        if G.has_edge(s, t):

            G[s][t]["weight"] += w

        else:

            G.add_edge(
                s,
                t,
                weight=w
            )

    return G

# ==========================================================
# DRAW NETWORK PANEL
# ==========================================================

def draw_panel(ax, G, title, panel_label):

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.85,
        iterations=500,
        weight="weight"
    )

    # node degree
    degrees = dict(
        G.degree(weight="weight")
    )

    # node sizes
    node_sizes = [

        120 + (degrees[n] ** 0.62) * 22

        for n in G.nodes()
    ]

    # node colors
    node_colors = [

        TYPE_COLORS.get(
            G.nodes[n].get("node_type", "OTHER"),
            TYPE_COLORS["OTHER"]
        )

        for n in G.nodes()
    ]

    # edge widths
    edge_weights = [

        G[u][v]["weight"]

        for u, v in G.edges()
    ]

    max_w = max(edge_weights) if edge_weights else 1

    edge_widths = [

        0.15 + (w / max_w) * 1.6

        for w in edge_weights
    ]

    # EDGES
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.18,
        edge_color="gray"
    )

    # NODES
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="white"
    )

    # LABELS FOR ALL NODES
    labels = {

        n: clean_label(n)

        for n in G.nodes()
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=6.4,
        font_family="DejaVu Sans",
        font_weight="bold"
    )

    # TITLE
    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    # PANEL LABEL
    ax.text(
        -0.04,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.axis("off")

# ==========================================================
# BUILD GRAPHS
# ==========================================================

G_dino = build_graph(
    DINO_EDGE_FILE,
    top_n=110
)

G_cyano = build_graph(
    CYANO_EDGE_FILE,
    top_n=110
)

# ==========================================================
# DRAW FIGURE
# ==========================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 18)
)

# PANEL A
draw_panel(
    axes[0],
    G_dino,
    "Dinoflagellate STX semantic knowledge graph",
    "A"
)

# PANEL B
draw_panel(
    axes[1],
    G_cyano,
    "Cyanobacterial STX semantic knowledge graph",
    "B"
)

# ==========================================================
# LEGEND
# ==========================================================

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
            label=LABEL_MAP.get(
                entity_type,
                entity_type
            ),
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

# ==========================================================
# MAIN TITLE
# ==========================================================

fig.suptitle(
    "Cross-taxa STX semantic knowledge graph structure",
    fontsize=17,
    fontweight="bold",
    y=0.995
)

# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout(
    rect=[0, 0, 0.86, 0.98]
)

# ==========================================================
# SAVE
# ==========================================================

plt.savefig(
    OUT_PNG,
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nDinoflagellate KG")
print("----------------------")
print("Nodes:", G_dino.number_of_nodes())
print("Edges:", G_dino.number_of_edges())

print("\nCyanobacterial KG")
print("----------------------")
print("Nodes:", G_cyano.number_of_nodes())
print("Edges:", G_cyano.number_of_edges())
