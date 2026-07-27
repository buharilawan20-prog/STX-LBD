import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

AI_FILE = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_dino_pre2016_vs_post2015_validated_unvalidated_refined.png"
OUT_PDF = OUT_DIR / "Figure_dino_pre2016_vs_post2015_validated_unvalidated_refined.pdf"

TOP_VALIDATED = 8
TOP_UNVALIDATED = 8
BRIDGES_PER_HYPOTHESIS = 2

HUB_BLACKLIST = {
    "alexandrium",
    "dinoflagellate",
    "dinoflagellates",
    "saxitoxin",
    "paralytic_shellfish_toxins",
    "paralytic_shellfish_poisoning",
    "biosynthesis",
    "toxicity",
    "cyanobacteria",
    "cyanobacterial",
    "cyanobacterium",
    "anabaena",
    "cylindrospermopsis",
    "dolichospermum"
}

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

def clean_label(x):
    x = str(x).replace("_", " ")
    x = x.replace("sxta", "sxtA")
    x = x.replace("sxtg", "sxtG")
    x = x.replace("sxtd", "sxtD")
    x = x.replace("sxti", "sxtI")
    x = x.replace("sxt genes", "sxt genes")
    x = x.replace("saxitoxin biosynthesis", "STX biosynthesis")
    return x

def is_bad_bridge(x):
    x = str(x).strip().lower()
    if x in HUB_BLACKLIST:
        return True
    if "cyanobacter" in x:
        return True
    return False

def node_color(node_type):
    return COLOR_MAP.get(node_type, COLOR_MAP["OTHER"])

def add_hypotheses_to_graph(G, df, validated_value):
    for _, row in df.iterrows():

        s = str(row["Source"]).strip()
        t = str(row["Target"]).strip()

        if not s or not t:
            continue

        s_type = str(row["Source_Type"]).strip()
        t_type = str(row["Target_Type"]).strip()

        if "CYANO" in s_type.upper() or "CYANO" in t_type.upper():
            continue

        score = float(row.get("Final_AI_Rank_Score", 0))

        G.add_node(s, node_type=s_type)
        G.add_node(t, node_type=t_type)

        G.add_edge(
            s,
            t,
            weight=score,
            validated=validated_value,
            edge_kind="hypothesis"
        )

        bridges = [
            b.strip()
            for b in str(row.get("Bridge_Nodes", "")).split(";")
            if b.strip()
        ]

        kept = 0

        for b in bridges:

            if kept >= BRIDGES_PER_HYPOTHESIS:
                break

            if is_bad_bridge(b):
                continue

            if b in [s, t]:
                continue

            G.add_node(b, node_type="OTHER")

            G.add_edge(
                s,
                b,
                weight=score * 0.25,
                validated=validated_value,
                edge_kind="bridge"
            )

            G.add_edge(
                b,
                t,
                weight=score * 0.25,
                validated=validated_value,
                edge_kind="bridge"
            )

            kept += 1

def draw_panel(ax, G, title, panel_label):
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.25,
        iterations=500,
        weight="weight"
    )

    node_types = nx.get_node_attributes(G, "node_type")
    degrees = dict(G.degree())

    node_colors = [
        node_color(node_types.get(n, "OTHER"))
        for n in G.nodes()
    ]

    node_sizes = [
        650 + degrees[n] * 120
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
        ax=ax,
        width=2.2,
        edge_color="gray",
        style="solid",
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=unvalidated_edges,
        ax=ax,
        width=2.2,
        edge_color="gray",
        style="dashed",
        alpha=0.8
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.9,
        alpha=0.95
    )

    labels = {n: clean_label(n) for n in G.nodes()}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold"
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    ax.text(
        -0.03,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.axis("off")

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

# Remove cyano source/target hypotheses for dinoflagellate-only figure
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

# ===============================
# BUILD TWO PANEL GRAPHS
# ===============================

G_validated = nx.Graph()
G_unvalidated = nx.Graph()

add_hypotheses_to_graph(G_validated, validated, 1)
add_hypotheses_to_graph(G_unvalidated, unvalidated, 0)

# ===============================
# PLOT
# ===============================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 16)
)

draw_panel(
    axes[0],
    G_validated,
    "Validated dinoflagellate STX hypotheses",
    "A"
)

draw_panel(
    axes[1],
    G_unvalidated,
    "Top unvalidated dinoflagellate STX predictions",
    "B"
)

# ===============================
# LEGEND
# ===============================

legend_nodes = []

for t, color in COLOR_MAP.items():
    legend_nodes.append(
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=LABEL_MAP.get(t, t),
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10
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

fig.legend(
    handles=legend_nodes + legend_edges,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=4,
    frameon=False,
    fontsize=11
)

fig.suptitle(
    "Dinoflagellate STX hypotheses: pre-2016 predictions evaluated against post-2015 literature",
    fontsize=18,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])

plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

plt.close()

# ===============================
# SAVE DATA USED
# ===============================

used = pd.concat(
    [
        validated.assign(Figure_Panel="A_validated"),
        unvalidated.assign(Figure_Panel="B_unvalidated")
    ],
    ignore_index=True
)

used.to_csv(
    OUT_DIR / "top_dino_validated_unvalidated_hypotheses_refined_used.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)

print("\nValidated panel:")
print("Nodes:", G_validated.number_of_nodes())
print("Edges:", G_validated.number_of_edges())

print("\nUnvalidated panel:")
print("Nodes:", G_unvalidated.number_of_nodes())
print("Edges:", G_unvalidated.number_of_edges())
