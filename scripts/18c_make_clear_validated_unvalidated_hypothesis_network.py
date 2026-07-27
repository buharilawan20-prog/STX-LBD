import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_clear_validated_unvalidated_AI_hypothesis_network.png"
OUT_PDF = OUT_DIR / "Figure_clear_validated_unvalidated_AI_hypothesis_network.pdf"
OUT_TABLE = OUT_DIR / "Figure_clear_validated_unvalidated_AI_hypothesis_network_table.csv"

TOP_VALIDATED = 12
TOP_UNVALIDATED = 12

TYPE_COLORS = {
    "TOXIN": "#D55E00",
    "SXT_GENE": "#0072B2",
    "DINO_TAXON": "#009E73",
    "ENV_FACTOR": "#E69F00",
    "BIOLOGICAL_PROCESS": "#CC79A7",
    "DETECTION_METHOD": "#999999",
    "OTHER": "#BBBBBB"
}

LABEL_MAP = {
    "TOXIN": "Toxin",
    "SXT_GENE": "sxt Gene",
    "DINO_TAXON": "Dino Taxon",
    "ENV_FACTOR": "Environmental Factor",
    "BIOLOGICAL_PROCESS": "Biological Process",
    "DETECTION_METHOD": "Detection Method",
    "OTHER": "Other"
}

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

def norm(x):
    return str(x).strip().lower().replace(" ", "_")

def clean_label(x):
    x = norm(x)

    repl = {
        "sxta": "sxtA",
        "sxtb": "sxtB",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth": "sxtH",
        "sxts": "sxtS",
        "sxt_genes": "sxt genes",
        "gonyautoxin": "GTX",
        "gtx": "GTX",
        "neosaxitoxin": "neoSTX",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "alexandrium_catenella": "Alexandrium catenella",
        "alexandrium_tamarense": "Alexandrium tamarense",
        "alexandrium_minutum": "Alexandrium minutum",
        "alexandrium_fundyense": "Alexandrium fundyense",
        "alexandrium_tamiyavanichii": "Alexandrium tamiyavanichii",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
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

def get_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column among: {options}")

def get_score_col(df):
    for c in ["Final_AI_Rank_Score", "ML_Probability", "Node2Vec_Integrated_Score", "Score"]:
        if c in df.columns:
            return c
    raise ValueError("No score column found.")

def make_graph(sub, source_col, target_col, source_type_col, target_type_col, score_col):
    G = nx.Graph()

    for _, r in sub.iterrows():
        s = norm(r[source_col])
        t = norm(r[target_col])

        if not s or not t or s == t:
            continue

        s_type = str(r[source_type_col]).strip()
        t_type = str(r[target_type_col]).strip()

        score = float(r[score_col])

        G.add_node(s, node_type=s_type)
        G.add_node(t, node_type=t_type)

        G.add_edge(s, t, weight=score)

    return G

def draw_graph(ax, G, title, edge_color):
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges after filtering", ha="center", va="center")
        ax.axis("off")
        return

    try:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=0.65, iterations=1000, weight="weight")

    degrees = dict(G.degree())

    node_sizes = [
        900 + degrees[n] * 350
        for n in G.nodes()
    ]

    node_colors = [
        TYPE_COLORS.get(G.nodes[n].get("node_type", "OTHER"), TYPE_COLORS["OTHER"])
        for n in G.nodes()
    ]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1

    edge_widths = [
        2.5 + (G[u][v]["weight"] / max_w) * 5
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_color,
        alpha=0.82
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.0,
        alpha=0.98
    )

    labels = {n: clean_label(n) for n in G.nodes()}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_family="DejaVu Sans",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=1.5
        )
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.axis("off")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT).fillna("")

source_col = get_col(df, ["Source", "source"])
target_col = get_col(df, ["Target", "target"])
source_type_col = get_col(df, ["Source_Type", "source_type"])
target_type_col = get_col(df, ["Target_Type", "target_type"])
label_col = get_col(df, ["Temporal_Label", "temporal_label"])
score_col = get_score_col(df)

df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

# Remove cyano nodes for dinoflagellate hypothesis figure
df = df[
    ~df[source_type_col].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

df = df[
    ~df[target_type_col].astype(str).str.contains("CYANO", case=False, na=False)
].copy()

# Remove only very generic hubs
df_filtered = df[
    ~df[source_col].apply(is_generic) &
    ~df[target_col].apply(is_generic)
].copy()

# fallback if too few after filtering
if len(df_filtered) < 10:
    df_filtered = df.copy()

validated = (
    df_filtered[df_filtered[label_col] == 1]
    .sort_values(score_col, ascending=False)
    .head(TOP_VALIDATED)
    .copy()
)

unvalidated = (
    df_filtered[df_filtered[label_col] == 0]
    .sort_values(score_col, ascending=False)
    .head(TOP_UNVALIDATED)
    .copy()
)

validated["Status"] = "Validated"
unvalidated["Status"] = "Unvalidated"

plot_df = pd.concat([validated, unvalidated], ignore_index=True)
plot_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

G_valid = make_graph(
    validated,
    source_col,
    target_col,
    source_type_col,
    target_type_col,
    score_col
)

G_unvalid = make_graph(
    unvalidated,
    source_col,
    target_col,
    source_type_col,
    target_type_col,
    score_col
)

# ==========================================================
# DRAW TWO-PANEL FIGURE
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

draw_graph(
    axes[0],
    G_valid,
    "A. Validated AI-ranked hypotheses\n(recovered in post-2015 literature)",
    "#2A9D8F"
)

draw_graph(
    axes[1],
    G_unvalid,
    "B. Unvalidated high-ranking predictions\n(candidate future hypotheses)",
    "#E76F51"
)

# ==========================================================
# LEGEND
# ==========================================================

used_types = set()

for G in [G_valid, G_unvalid]:
    used_types.update(nx.get_node_attributes(G, "node_type").values())

legend_handles = []

for t in sorted(used_types):
    legend_handles.append(
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

legend_handles.extend([
    Line2D([0], [0], color="#2A9D8F", lw=4, label="Validated edge"),
    Line2D([0], [0], color="#E76F51", lw=4, label="Unvalidated/predicted edge")
])

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.02)
)

fig.suptitle(
    "Top AI-ranked dinoflagellate STX hypotheses from pre-2016 knowledge",
    fontsize=20,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.925,
    "Validated hypotheses were recovered in post-2015 literature; unvalidated predictions represent candidate future biological links.",
    ha="center",
    fontsize=12
)

plt.tight_layout(rect=[0, 0.08, 1, 0.90])

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_TABLE)

print("\nValidated network:")
print("Nodes:", G_valid.number_of_nodes())
print("Edges:", G_valid.number_of_edges())

print("\nUnvalidated network:")
print("Nodes:", G_unvalid.number_of_nodes())
print("Edges:", G_unvalid.number_of_edges())

print("\nScore column:", score_col)
