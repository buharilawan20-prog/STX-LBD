import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_top_validated_unvalidated_AI_hypothesis_network.png"
OUT_PDF = OUT_DIR / "Figure_top_validated_unvalidated_AI_hypothesis_network.pdf"
OUT_TABLE = OUT_DIR / "Figure_top_validated_unvalidated_AI_hypothesis_network_table.csv"

# ==========================================================
# SETTINGS
# ==========================================================

TOP_VALIDATED = 10
TOP_UNVALIDATED = 10

REMOVE_GENERIC_HUBS = True

GENERIC_HUBS = {
    "saxitoxin",
    "stx",
    "paralytic_shellfish_toxins",
    "paralytic_shellfish_poisoning",
    "dinoflagellate",
    "dinoflagellates",
    "cyanobacteria",
    "cyanobacterial",
    "cyanobacterium",
    "toxin",
    "toxins",
    "toxicity"
}

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

STATUS_COLORS = {
    "Validated": "#2A9D8F",
    "Unvalidated": "#E76F51"
}

STATUS_STYLES = {
    "Validated": "solid",
    "Unvalidated": "dashed"
}

# ==========================================================
# HELPERS
# ==========================================================

def norm(x):
    return str(x).strip().lower().replace(" ", "_")

def clean_label(x):
    x = norm(x)

    repl = {
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth": "sxtH",
        "sxts": "sxtS",
        "saxitoxin": "STX",
        "gonyautoxin": "GTX",
        "neosaxitoxin": "neoSTX",
        "paralytic_shellfish_toxins": "PSTs",
        "paralytic_shellfish_poisoning": "PSP",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
        "alexandrium_catenella": "Alexandrium catenella",
        "alexandrium_tamarense": "Alexandrium tamarense",
        "alexandrium_minutum": "Alexandrium minutum",
        "alexandrium_fundyense": "Alexandrium fundyense",
        "pyrodinium_bahamense": "Pyrodinium bahamense",
        "mass_spectrometry": "mass spectrometry",
        "mouse_bioassay": "mouse bioassay",
        "lc_ms": "LC-MS",
        "hplc": "HPLC",
        "elisa": "ELISA"
    }

    return repl.get(x, x.replace("_", " "))

def is_generic(x):
    return norm(x) in GENERIC_HUBS

def get_score_column(df):
    for c in [
        "Final_AI_Rank_Score",
        "ML_Probability",
        "Node2Vec_Integrated_Score",
        "Score"
    ]:
        if c in df.columns:
            return c
    raise ValueError("No score column found.")

def get_required_column(df, names):
    for c in names:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column. Tried: {names}")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT).fillna("")

source_col = get_required_column(df, ["Source", "source"])
target_col = get_required_column(df, ["Target", "target"])
source_type_col = get_required_column(df, ["Source_Type", "source_type"])
target_type_col = get_required_column(df, ["Target_Type", "target_type"])
label_col = get_required_column(df, ["Temporal_Label", "temporal_label"])
score_col = get_score_column(df)

df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

# Dinoflagellate hypothesis figure only: remove cyano source/target types
df = df[
    ~df[source_type_col].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

df = df[
    ~df[target_type_col].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

if REMOVE_GENERIC_HUBS:
    df = df[
        ~df[source_col].apply(is_generic) &
        ~df[target_col].apply(is_generic)
    ].copy()

validated = (
    df[df[label_col] == 1]
    .sort_values(score_col, ascending=False)
    .head(TOP_VALIDATED)
)

unvalidated = (
    df[df[label_col] == 0]
    .sort_values(score_col, ascending=False)
    .head(TOP_UNVALIDATED)
)

plot_df = pd.concat(
    [
        validated.assign(Status="Validated"),
        unvalidated.assign(Status="Unvalidated")
    ],
    ignore_index=True
)

if plot_df.empty:
    raise ValueError("No hypotheses available after filtering. Try REMOVE_GENERIC_HUBS=False.")

plot_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

# ==========================================================
# BUILD NETWORK
# ==========================================================

G = nx.Graph()

for _, r in plot_df.iterrows():

    s = norm(r[source_col])
    t = norm(r[target_col])

    s_type = str(r[source_type_col]).strip()
    t_type = str(r[target_type_col]).strip()

    status = r["Status"]
    score = float(r[score_col])

    if not s or not t or s == t:
        continue

    G.add_node(s, node_type=s_type)
    G.add_node(t, node_type=t_type)

    G.add_edge(
        s,
        t,
        weight=score,
        status=status
    )

# ==========================================================
# DRAW NETWORK
# ==========================================================

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.1,
    iterations=700,
    weight="weight"
)

degrees = dict(G.degree())

node_sizes = [
    700 + degrees[n] * 220
    for n in G.nodes()
]

node_colors = [
    TYPE_COLORS.get(G.nodes[n].get("node_type", "OTHER"), TYPE_COLORS["OTHER"])
    for n in G.nodes()
]

# Draw edges by status
for status in ["Validated", "Unvalidated"]:

    edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("status") == status
    ]

    widths = [
        1.5 + G[u][v]["weight"] * 3
        for u, v in edges
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=widths,
        edge_color=STATUS_COLORS[status],
        style=STATUS_STYLES[status],
        alpha=0.75
    )

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="black",
    linewidths=0.8,
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
    font_weight="bold",
    font_family="DejaVu Sans"
)

# ==========================================================
# LEGEND
# ==========================================================

entity_legend = []

used_types = sorted(set(nx.get_node_attributes(G, "node_type").values()))

for t in used_types:
    entity_legend.append(
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=LABEL_MAP.get(t, t),
            markerfacecolor=TYPE_COLORS.get(t, TYPE_COLORS["OTHER"]),
            markeredgecolor="black",
            markersize=10
        )
    )

status_legend = [
    Line2D(
        [0], [0],
        color=STATUS_COLORS["Validated"],
        lw=2.5,
        linestyle="solid",
        label="Validated in post-2015"
    ),
    Line2D(
        [0], [0],
        color=STATUS_COLORS["Unvalidated"],
        lw=2.5,
        linestyle="dashed",
        label="Unvalidated / predicted"
    )
]

plt.legend(
    handles=entity_legend + status_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=4,
    frameon=False,
    fontsize=10
)

plt.title(
    "Top AI-ranked dinoflagellate STX hypotheses validated against post-2015 literature",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.axis("off")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_TABLE)

print("\nNetwork summary:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Validated plotted:", len(validated))
print("Unvalidated plotted:", len(unvalidated))
print("Score column:", score_col)
