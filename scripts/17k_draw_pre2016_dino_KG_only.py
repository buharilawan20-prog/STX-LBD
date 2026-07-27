import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_pre2016_semantic_edges.csv"
OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_pre2016_dinoflagellate_KG.png"
OUT_PDF = OUT_DIR / "Figure_pre2016_dinoflagellate_KG.pdf"

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

LABEL_MAP = {
    "TOXIN": "Toxin",
    "SXT_GENE": "sxt Gene",
    "DINO_TAXON": "Dino Taxon",
    "CYANO_TAXON": "Cyano Taxon",
    "ENV_FACTOR": "Environmental Factor",
    "BIOLOGICAL_PROCESS": "Biological Process",
    "DETECTION_METHOD": "Detection Method",
    "OTHER": "Other"
}

def clean_label(x):
    x = str(x)
    replacements = {
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "paralytic_shellfish_toxins": "paralytic shellfish toxins",
        "paralytic_shellfish_poisoning": "paralytic shellfish poisoning",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
        "pyrodinium_bahamense": "Pyrodinium bahamense",
        "toxin_production": "toxin production",
        "harmful_algal_bloom": "harmful algal bloom",
        "mouse_bioassay": "mouse bioassay",
        "mass_spectrometry": "mass spectrometry",
        "lc_ms": "LC-MS",
        "hplc": "HPLC",
        "gtx": "GTX"
    }
    return replacements.get(x, x.replace("_", " "))

df = pd.read_csv(EDGE_FILE).fillna("")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

# Keep strongest edges for readable figure
df = df.sort_values("weight", ascending=False).head(120)

G = nx.Graph()

for _, r in df.iterrows():
    s = str(r["source"]).strip()
    t = str(r["target"]).strip()

    if not s or not t or s == t:
        continue

    G.add_node(s, node_type=str(r.get("source_type", "OTHER")))
    G.add_node(t, node_type=str(r.get("target_type", "OTHER")))

    G.add_edge(s, t, weight=float(r["weight"]))

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.95,
    iterations=600,
    weight="weight"
)

degrees = dict(G.degree(weight="weight"))

node_sizes = [
    160 + (degrees[n] ** 0.65) * 30
    for n in G.nodes()
]

node_colors = [
    TYPE_COLORS.get(G.nodes[n].get("node_type", "OTHER"), TYPE_COLORS["OTHER"])
    for n in G.nodes()
]

edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_w = max(edge_weights) if edge_weights else 1

edge_widths = [
    0.2 + (w / max_w) * 2.0
    for w in edge_weights
]

plt.figure(figsize=(12, 10))

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    edge_color="gray",
    alpha=0.22
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="white",
    linewidths=0.6,
    alpha=0.95
)

labels = {n: clean_label(n) for n in G.nodes()}

nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=7,
    font_weight="bold",
    font_family="DejaVu Sans"
)

legend_items = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=LABEL_MAP[k],
        markerfacecolor=v,
        markersize=9
    )
    for k, v in TYPE_COLORS.items()
    if k != "OTHER"
]

plt.legend(
    handles=legend_items,
    title="Entity type",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False
)

plt.title(
    "Pre-2016 dinoflagellate STX semantic knowledge graph",
    fontsize=18,
    fontweight="bold",
    pad=18
)

plt.axis("off")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nKG summary:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
