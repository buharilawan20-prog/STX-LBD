import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUTDIR = BASE / "FINAL_WORKSPACE/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure_post2015_dino_KG.png"
OUT_PDF = OUTDIR / "Figure_post2015_dino_KG.pdf"

TOP_N_NODES = 35
SEED = 42

TYPE_COLORS = {
    "TOXIN": "#E66101",
    "SXT_GENE": "#1F78B4",
    "DINO_TAXON": "#1B9E77",
    "CYANO_TAXON": "#67C5DF",
    "ENV_FACTOR": "#E6A600",
    "BIOLOGICAL_PROCESS": "#CC79A7",
    "DETECTION_METHOD": "#9E9E9E",
}

TYPE_LABELS = {
    "TOXIN": "Toxin",
    "SXT_GENE": "sxt gene",
    "DINO_TAXON": "Dinoflagellate taxon",
    "CYANO_TAXON": "Cyanobacteria taxon",
    "ENV_FACTOR": "Environmental factor",
    "BIOLOGICAL_PROCESS": "Biological process",
    "DETECTION_METHOD": "Detection method",
}

def clean_label(x):
    x = str(x).strip()
    mapping = {
        "paralytic_shellfish_toxins": "paralytic\nshellfish\ntoxins",
        "paralytic_shellfish_poisoning": "paralytic\nshellfish\npoisoning",
        "toxin_biosynthesis": "toxin\nbiosynthesis",
        "gene_expression": "gene\nexpression",
        "gymnodinium_catenatum": "Gymnodinium\ncatenatum",
        "pyrodinium_bahamense": "Pyrodinium\nbahamense",
        "alexandrium_catenella": "A. catenella",
        "alexandrium_minutum": "A. minutum",
        "alexandrium_tamarense": "A. tamarense",
        "alexandrium_fundyense": "A. fundyense",
        "alexandrium_pacificum": "A. pacificum",
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxtb": "sxtB",
        "sxt_genes": "sxt genes",
        "gtx": "GTX",
        "hplc": "HPLC",
        "lc_ms": "LC-MS",
    }
    return mapping.get(x.lower(), x.replace("_", " "))

df = pd.read_csv(EDGE_FILE).fillna("")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

G = nx.Graph()

for _, r in df.iterrows():
    s = str(r["source"]).strip()
    t = str(r["target"]).strip()

    if not s or not t or s == t:
        continue

    G.add_node(s, entity_type=str(r["source_type"]).strip())
    G.add_node(t, entity_type=str(r["target_type"]).strip())

    if G.has_edge(s, t):
        G[s][t]["weight"] += float(r["weight"])
    else:
        G.add_edge(s, t, weight=float(r["weight"]))

print("Full graph")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

weighted_degree = dict(G.degree(weight="weight"))

top_nodes = sorted(
    weighted_degree,
    key=weighted_degree.get,
    reverse=True
)[:TOP_N_NODES]

H = G.subgraph(top_nodes).copy()

print("Plotted graph")
print("Nodes:", H.number_of_nodes())
print("Edges:", H.number_of_edges())

pos = nx.spring_layout(
    H,
    seed=SEED,
    k=0.75,
    iterations=300,
    weight="weight"
)

deg = dict(H.degree(weight="weight"))
deg_vals = np.array(list(deg.values()), dtype=float)

if deg_vals.max() > deg_vals.min():
    node_sizes = {
        n: 250 + 2300 * ((deg[n] - deg_vals.min()) / (deg_vals.max() - deg_vals.min()))
        for n in H.nodes()
    }
else:
    node_sizes = {n: 800 for n in H.nodes()}

edge_weights = np.array(
    [H[u][v]["weight"] for u, v in H.edges()],
    dtype=float
)

if len(edge_weights) and edge_weights.max() > edge_weights.min():
    edge_widths = [
        0.25 + 1.8 * ((H[u][v]["weight"] - edge_weights.min()) / (edge_weights.max() - edge_weights.min()))
        for u, v in H.edges()
    ]
else:
    edge_widths = [0.6 for _ in H.edges()]

plt.rcParams["font.family"] = "DejaVu Serif"

fig, ax = plt.subplots(figsize=(11, 8.5))

nx.draw_networkx_edges(
    H,
    pos,
    ax=ax,
    width=edge_widths,
    edge_color="#9E9E9E",
    alpha=0.35
)

for etype, color in TYPE_COLORS.items():
    nodes = [
        n for n, d in H.nodes(data=True)
        if d.get("entity_type") == etype
    ]

    if nodes:
        nx.draw_networkx_nodes(
            H,
            pos,
            nodelist=nodes,
            node_size=[node_sizes[n] for n in nodes],
            node_color=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.96,
            ax=ax
        )

labels = {n: clean_label(n) for n in H.nodes()}

nx.draw_networkx_labels(
    H,
    pos,
    labels=labels,
    font_size=8.5,
    font_family="DejaVu Serif",
    ax=ax
)

legend_order = [
    "TOXIN",
    "SXT_GENE",
    "DINO_TAXON",
    "CYANO_TAXON",
    "ENV_FACTOR",
    "BIOLOGICAL_PROCESS",
    "DETECTION_METHOD",
]

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=TYPE_LABELS[t],
        markerfacecolor=TYPE_COLORS[t],
        markeredgecolor="black",
        markersize=9
    )
    for t in legend_order
]

legend = ax.legend(
    handles=legend_elements,
    title="Entity type",
    loc="lower left",
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    ncol=2
)

legend.get_title().set_fontweight("bold")

ax.set_title(
    "Post-2015 Dinoflagellate Semantic Knowledge Graph",
    fontsize=16,
    fontweight="bold",
    pad=16
)
 
ax.axis("off")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=500, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
