import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")
IN_DIR = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

validated_file = IN_DIR / "top_validated_hypotheses_evidence_weighted.csv"
unvalidated_file = IN_DIR / "top_unvalidated_hypotheses_refined.csv"

validated = pd.read_csv(validated_file)
unvalidated = pd.read_csv(unvalidated_file)

validated["status"] = "Validated"
unvalidated["status"] = "Unvalidated"

# Top hypotheses
validated = validated.head(15)
unvalidated = unvalidated.head(15)

df = pd.concat([validated, unvalidated], ignore_index=True)

print(df.columns.tolist())

# -------------------------------------------------------
# Parse source and target from Hypothesis column
# -------------------------------------------------------

def split_hypothesis(text):

    text = str(text)

    for sep in ["↔", "-", "–", "—"]:

        if sep in text:

            parts = text.split(sep, 1)

            return parts[0].strip(), parts[1].strip()

    return text, text

df[["source", "target"]] = df["Hypothesis"].apply(
    lambda x: pd.Series(split_hypothesis(x))
)

# -------------------------------------------------------
# Build network
# -------------------------------------------------------

G = nx.Graph()

for _, row in df.iterrows():

    source = row["source"]
    target = row["target"]

    G.add_node(source)
    G.add_node(target)

    G.add_edge(
        source,
        target,
        status=row["status"],
        score=row.get("AI_Probability", 1.0),
        category=row.get("Biological_Category", "Unknown")
    )

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -------------------------------------------------------
# Layout
# -------------------------------------------------------

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.8,
    iterations=500
)

plt.figure(figsize=(12,10))

validated_edges = [
    (u,v)
    for u,v,d in G.edges(data=True)
    if d["status"]=="Validated"
]

unvalidated_edges = [
    (u,v)
    for u,v,d in G.edges(data=True)
    if d["status"]=="Unvalidated"
]

# validated
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=validated_edges,
    edge_color="forestgreen",
    width=2.5,
    alpha=0.8
)

# unvalidated
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=unvalidated_edges,
    edge_color="crimson",
    width=2.0,
    style="dashed",
    alpha=0.8
)

degrees = dict(G.degree())

node_sizes = [
    500 + degrees[n]*120
    for n in G.nodes()
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color="lightsteelblue",
    edgecolors="black",
    linewidths=0.8
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title(
    "Top Validated and Unvalidated STX-LBD Hypotheses",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()

png_file = OUT_DIR / "validated_unvalidated_hypothesis_network.png"
pdf_file = OUT_DIR / "validated_unvalidated_hypothesis_network.pdf"

plt.savefig(png_file, dpi=600, bbox_inches="tight")
plt.savefig(pdf_file, bbox_inches="tight")

print("\nSaved:")
print(png_file)
print(pdf_file)
