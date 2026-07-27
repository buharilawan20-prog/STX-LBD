import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

AI_FILE = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_one_network_dino_validated_unvalidated_hypotheses.png"
OUT_PDF = OUT_DIR / "Figure_one_network_dino_validated_unvalidated_hypotheses.pdf"
OUT_DATA = OUT_DIR / "one_network_dino_validated_unvalidated_hypotheses_used.csv"

TOP_VALIDATED = 10
TOP_UNVALIDATED = 10
BRIDGES_PER_HYPOTHESIS = 2

# ===============================
# REMOVE OVER-GENERIC BRIDGES
# ===============================

HUB_BLACKLIST = {
    "saxitoxin",
    "stx",
    "paralytic_shellfish_toxins",
    "paralytic shellfish toxins",
    "paralytic_shellfish_poisoning",
    "paralytic shellfish poisoning",
    "dinoflagellate",
    "dinoflagellates",
    "alexandrium",
    "toxicity",
    "biosynthesis",
    "cyanobacteria",
    "cyanobacterial",
    "cyanobacterium",
    "anabaena",
    "cylindrospermopsis",
    "dolichospermum",
    "aphanizomenon"
}

# ===============================
# COLORS
# ===============================

COLOR_MAP = {
    "TOXIN": "#6E6E6E",
    "SXT_GENE": "#1F77B4",
    "ENV_FACTOR": "#D62728",
    "BIOLOGICAL_PROCESS": "#2CA02C",
    "DINO_TAXON": "#9467BD",
    "OTHER": "#BDBDBD"
}

LABEL_MAP = {
    "TOXIN": "TOXIN",
    "SXT_GENE": "GENE",
    "ENV_FACTOR": "ENV",
    "BIOLOGICAL_PROCESS": "MECHANISM",
    "DINO_TAXON": "TAXON",
    "OTHER": "OTHER"
}

# ===============================
# FUNCTIONS
# ===============================

def normalize_node(x):
    x = str(x).strip().lower()
    x = x.replace(" ", "_")

    mapping = {
        "stx": "saxitoxin",
        "saxitoxins": "saxitoxin",
        "pst": "paralytic_shellfish_toxins",
        "psts": "paralytic_shellfish_toxins",
        "paralytic_shellfish_toxin": "paralytic_shellfish_toxins",
        "paralytic_shellfish_toxins": "paralytic_shellfish_toxins",
        "paralytic_shellfish_poisoning": "paralytic_shellfish_poisoning",

        "sxta1": "sxta",
        "sxta4": "sxta",
        "sxta_domain": "sxta",
        "sxt_gene": "sxt_genes",
        "sxt_genes": "sxt_genes",

        "dinoflagellates": "dinoflagellate",
        "cyanobacterial": "cyanobacteria",
        "cyanobacterium": "cyanobacteria",

        "gymnodinium_catenatum": "gymnodinium_catenatum",
        "alexandrium_catenella": "alexandrium_catenella",
        "alexandrium_tamarense": "alexandrium_tamarense",
        "alexandrium_minutum": "alexandrium_minutum",
        "alexandrium_fundyense": "alexandrium_fundyense",
        "pyrodinium_bahamense": "pyrodinium_bahamense",

        "saxitoxin_biosynthesis": "stx_biosynthesis",
        "toxin_biosynthesis": "toxin_biosynthesis",
        "toxin_production": "toxin_production",
        "gene_expression": "gene_expression",
        "mass_spectrometry": "mass_spectrometry",
        "lc_ms": "lc_ms",
        "lc_ms_ms": "lc_ms_ms"
    }

    return mapping.get(x, x)


def clean_label(x):
    x = str(x)

    label_map = {
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxt_genes": "sxt genes",
        "stx_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "paralytic_shellfish_toxins": "paralytic shellfish toxins",
        "paralytic_shellfish_poisoning": "paralytic shellfish poisoning",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
        "alexandrium_catenella": "Alexandrium catenella",
        "alexandrium_tamarense": "Alexandrium tamarense",
        "alexandrium_minutum": "Alexandrium minutum",
        "alexandrium_fundyense": "Alexandrium fundyense",
        "pyrodinium_bahamense": "Pyrodinium bahamense",
        "mass_spectrometry": "mass spectrometry",
        "lc_ms": "LC-MS",
        "hplc": "HPLC"
    }

    return label_map.get(x, x.replace("_", " "))


def is_bad_node(x):
    x = normalize_node(x)
    if x in HUB_BLACKLIST:
        return True
    if "cyanobacter" in x:
        return True
    return False


def infer_bridge_type(node):
    node = normalize_node(node)

    if node.startswith("sxt") or node in {"sxta", "sxtg", "sxtd", "sxti", "sxt_genes"}:
        return "SXT_GENE"

    if node in {"light", "temperature", "salinity", "nitrate", "nitrogen", "phosphorus", "phosphate", "nutrient", "warming", "bloom"}:
        return "ENV_FACTOR"

    if node in {"gonyautoxin", "neosaxitoxin", "gtx", "saxitoxin", "paralytic_shellfish_toxins"}:
        return "TOXIN"

    if node.startswith("alexandrium") or node.startswith("gymnodinium") or node.startswith("pyrodinium"):
        return "DINO_TAXON"

    if node in {"evolution", "regulation", "expression", "stx_biosynthesis", "toxin_biosynthesis", "toxin_production", "phylogenetic"}:
        return "BIOLOGICAL_PROCESS"

    return "OTHER"


def node_color(node_type):
    return COLOR_MAP.get(node_type, COLOR_MAP["OTHER"])


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(AI_FILE).fillna("")

df["Temporal_Label"] = pd.to_numeric(
    df["Temporal_Label"],
    errors="coerce"
).fillna(0).astype(int)

df["Final_AI_Rank_Score"] = pd.to_numeric(
    df["Final_AI_Rank_Score"],
    errors="coerce"
).fillna(0)

# Dinoflagellate-only: remove cyano source/target hypotheses
df = df[
    ~df["Source_Type"].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

df = df[
    ~df["Target_Type"].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

validated = df[df["Temporal_Label"] == 1].sort_values(
    "Final_AI_Rank_Score",
    ascending=False
).head(TOP_VALIDATED)

unvalidated = df[df["Temporal_Label"] == 0].sort_values(
    "Final_AI_Rank_Score",
    ascending=False
).head(TOP_UNVALIDATED)

plot_df = pd.concat([validated, unvalidated], ignore_index=True)

# ===============================
# BUILD ONE NETWORK
# ===============================

G = nx.Graph()

for _, row in plot_df.iterrows():

    s = normalize_node(row["Source"])
    t = normalize_node(row["Target"])

    if not s or not t:
        continue

    if is_bad_node(s) or is_bad_node(t):
        continue

    s_type = str(row["Source_Type"]).strip()
    t_type = str(row["Target_Type"]).strip()

    validated_status = int(row["Temporal_Label"])
    score = float(row["Final_AI_Rank_Score"])

    G.add_node(s, node_type=s_type)
    G.add_node(t, node_type=t_type)

    G.add_edge(
        s,
        t,
        weight=score,
        validated=validated_status,
        edge_kind="hypothesis"
    )

    bridges = [
        normalize_node(b.strip())
        for b in str(row.get("Bridge_Nodes", "")).split(";")
        if b.strip()
    ]

    kept = 0

    for b in bridges:

        if kept >= BRIDGES_PER_HYPOTHESIS:
            break

        if not b or b in {s, t}:
            continue

        if is_bad_node(b):
            continue

        b_type = infer_bridge_type(b)

        G.add_node(b, node_type=b_type)

        if not G.has_edge(s, b):
            G.add_edge(
                s,
                b,
                weight=score * 0.25,
                validated=validated_status,
                edge_kind="bridge"
            )

        if not G.has_edge(b, t):
            G.add_edge(
                b,
                t,
                weight=score * 0.25,
                validated=validated_status,
                edge_kind="bridge"
            )

        kept += 1

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(16, 11))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.15,
    iterations=600,
    weight="weight"
)

node_types = nx.get_node_attributes(G, "node_type")
degrees = dict(G.degree())

node_colors = [
    node_color(node_types.get(n, "OTHER"))
    for n in G.nodes()
]

node_sizes = [
    700 + degrees[n] * 130
    for n in G.nodes()
]

validated_edges = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get("validated", 0) == 1
]

unvalidated_edges = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get("validated", 0) == 0
]

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=validated_edges,
    width=2.3,
    edge_color="gray",
    style="solid",
    alpha=0.85
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=unvalidated_edges,
    width=2.3,
    edge_color="gray",
    style="dashed",
    alpha=0.75
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="black",
    linewidths=1.0,
    alpha=0.95
)

labels = {
    n: clean_label(n)
    for n in G.nodes()
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=9,
    font_weight="bold"
)

plt.title(
    "Top dinoflagellate STX hypotheses: validated and unvalidated pre-2016 predictions",
    fontsize=20,
    fontweight="bold",
    pad=20
)

plt.axis("off")

# ===============================
# LEGEND
# ===============================

legend_nodes = []

used_types = sorted(set(node_types.values()))

for t in used_types:
    legend_nodes.append(
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=LABEL_MAP.get(t, t),
            markerfacecolor=node_color(t),
            markeredgecolor="black",
            markersize=11
        )
    )

legend_edges = [
    Line2D(
        [0], [0],
        color="gray",
        lw=2.5,
        linestyle="-",
        label="Validated in post-2015 literature"
    ),
    Line2D(
        [0], [0],
        color="gray",
        lw=2.5,
        linestyle="--",
        label="Predicted / not yet validated"
    )
]

plt.legend(
    handles=legend_nodes + legend_edges,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=4,
    frameon=False,
    fontsize=11
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

plot_df.to_csv(
    OUT_DATA,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_DATA)

print("\nNetwork summary:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Validated hypotheses used:", len(validated))
print("Unvalidated hypotheses used:", len(unvalidated))
