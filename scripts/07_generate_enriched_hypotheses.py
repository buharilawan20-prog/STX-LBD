import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import combinations

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG_DIR = BASE / "FINAL_WORKSPACE/kg"
OUT_DIR = BASE / "FINAL_WORKSPACE/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_EDGE_FILE = KG_DIR / "dino_pre2016_semantic_edges.csv"

OUTFILE = OUT_DIR / "dino_pre2016_enriched_hypotheses.csv"

# ===============================
# SETTINGS
# ===============================

MIN_BRIDGE_SUPPORT = 1
TOP_N = 5000

INTERESTING_TYPES = {
    "SXT_GENE",
    "TOXIN",
    "ENV_FACTOR",
    "BIOLOGICAL_PROCESS",
    "DINO_TAXON",
    "CYANO_TAXON"
}

# ===============================
# LOAD EDGES
# ===============================

edges = pd.read_csv(INPUT_EDGE_FILE).fillna("")

required = [
    "source", "source_type",
    "target", "target_type",
    "relation", "weight"
]

for col in required:
    if col not in edges.columns:
        raise ValueError(f"Missing column: {col}")

edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1)

# ===============================
# BUILD GRAPH
# ===============================

G = nx.Graph()

node_types = {}

for _, row in edges.iterrows():

    s = str(row["source"]).strip()
    t = str(row["target"]).strip()

    if not s or not t or s == t:
        continue

    s_type = str(row["source_type"]).strip()
    t_type = str(row["target_type"]).strip()

    w = float(row["weight"])

    node_types[s] = s_type
    node_types[t] = t_type

    if G.has_edge(s, t):
        G[s][t]["weight"] += w
        G[s][t]["relations"].add(row["relation"])
    else:
        G.add_edge(
            s,
            t,
            weight=w,
            relations={row["relation"]}
        )

print("\nGraph loaded:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ===============================
# GENERATE CANDIDATES
# ===============================

hypotheses = []

nodes = [
    n for n in G.nodes()
    if node_types.get(n, "") in INTERESTING_TYPES
]

for source, target in combinations(nodes, 2):

    if source == target:
        continue

    if G.has_edge(source, target):
        continue

    source_type = node_types.get(source, "")
    target_type = node_types.get(target, "")

    # Avoid same-type generic low-value pairs
    if source_type == target_type and source_type in ["TOXIN", "DINO_TAXON", "CYANO_TAXON"]:
        continue

    common_neighbors = list(nx.common_neighbors(G, source, target))

    if len(common_neighbors) < MIN_BRIDGE_SUPPORT:
        continue

    bridge_rows = []

    bridge_score = 0

    for b in common_neighbors:

        b_type = node_types.get(b, "")

        w1 = G[source][b].get("weight", 1)
        w2 = G[target][b].get("weight", 1)

        local_score = min(w1, w2)

        bridge_score += local_score

        bridge_rows.append({
            "bridge": b,
            "bridge_type": b_type,
            "source_bridge_weight": w1,
            "target_bridge_weight": w2,
            "local_score": local_score
        })

    bridge_rows = sorted(
        bridge_rows,
        key=lambda x: x["local_score"],
        reverse=True
    )

    bridge_nodes = "; ".join([x["bridge"] for x in bridge_rows[:20]])
    bridge_types = "; ".join([x["bridge_type"] for x in bridge_rows[:20]])

    distinct_bridge_types = len(set(x["bridge_type"] for x in bridge_rows))

    degree_source = G.degree(source)
    degree_target = G.degree(target)

    # Adamic-Adar
    try:
        aa_score = list(nx.adamic_adar_index(G, [(source, target)]))[0][2]
    except Exception:
        aa_score = 0

    # Preferential attachment
    try:
        pa_score = list(nx.preferential_attachment(G, [(source, target)]))[0][2]
    except Exception:
        pa_score = degree_source * degree_target

    # Jaccard
    try:
        jaccard_score = list(nx.jaccard_coefficient(G, [(source, target)]))[0][2]
    except Exception:
        jaccard_score = 0

    score = (
        bridge_score * 2
        + len(common_neighbors) * 3
        + distinct_bridge_types * 5
        + aa_score * 2
        + jaccard_score * 10
    )

    # Biological hypothesis class
    pair_types = {source_type, target_type}

    if "ENV_FACTOR" in pair_types and "SXT_GENE" in pair_types:
        hyp_class = "environment_gene_regulation"
    elif "ENV_FACTOR" in pair_types and "TOXIN" in pair_types:
        hyp_class = "environment_toxin_association"
    elif "DINO_TAXON" in pair_types and "SXT_GENE" in pair_types:
        hyp_class = "taxon_gene_association"
    elif "DINO_TAXON" in pair_types and "TOXIN" in pair_types:
        hyp_class = "taxon_toxin_association"
    elif "BIOLOGICAL_PROCESS" in pair_types and "SXT_GENE" in pair_types:
        hyp_class = "gene_process_association"
    elif "BIOLOGICAL_PROCESS" in pair_types and "TOXIN" in pair_types:
        hyp_class = "process_toxin_association"
    elif "CYANO_TAXON" in pair_types and "DINO_TAXON" in pair_types:
        hyp_class = "cross_taxa_association"
    elif "CYANO_TAXON" in pair_types and "SXT_GENE" in pair_types:
        hyp_class = "cyano_gene_transfer_signal"
    else:
        hyp_class = "semantic_hypothesis"

    hypotheses.append({
        "Source": source,
        "Source_Type": source_type,
        "Target": target,
        "Target_Type": target_type,
        "Hypothesis_Class": hyp_class,
        "Score": score,
        "Bridge_Score": bridge_score,
        "Common_Neighbors": len(common_neighbors),
        "Distinct_Bridge_Types": distinct_bridge_types,
        "Adamic_Adar": aa_score,
        "Jaccard": jaccard_score,
        "Preferential_Attachment": pa_score,
        "Degree_Source": degree_source,
        "Degree_Target": degree_target,
        "Bridge_Nodes": bridge_nodes,
        "Bridge_Types": bridge_types,
        "Training_Graph": "dino_pre2016",
        "Candidate_Status": "unvalidated_candidate",
        "Embedding_Source_Target_Similarity": "",
        "Embedding_Bridge_Mean_Similarity": "",
        "Node2Vec_Integrated_Score": "",
        "ML_Probability": "",
        "Final_AI_Rank_Score": ""
    })

hyp_df = pd.DataFrame(hypotheses)

if len(hyp_df) == 0:
    print("No hypotheses generated.")
else:
    hyp_df = hyp_df.sort_values(
        by="Score",
        ascending=False
    ).head(TOP_N)

    hyp_df.to_csv(
        OUTFILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved hypotheses:")
    print(OUTFILE)

    print("\nTotal hypotheses:", len(hyp_df))

    print("\nHypothesis classes:")
    print(hyp_df["Hypothesis_Class"].value_counts())

    print("\nTop hypotheses:")
    print(
        hyp_df[
            [
                "Source",
                "Source_Type",
                "Target",
                "Target_Type",
                "Hypothesis_Class",
                "Score",
                "Common_Neighbors",
                "Bridge_Nodes"
            ]
        ].head(30).to_string(index=False)
    )
