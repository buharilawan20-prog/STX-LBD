from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pandas as pd
import networkx as nx


EDGE_FILE = "data/graphs/dino_pre2016_edges.csv"
OUTPUT_DIR = "results/hypotheses"
OUTPUT_FILE = "STX_BIOLOGY_HYPOTHESES.csv"


ALLOWED_PAIR_TYPES = {
    ("GENE", "TOXIN"): "GENE_TOXIN_HYPOTHESIS",
    ("GENE", "PROCESS"): "GENE_PROCESS_HYPOTHESIS",
    ("GENE", "SPECIES"): "GENE_SPECIES_HYPOTHESIS",
    ("ENV_FACTOR", "TOXIN"): "ENV_TOXIN_HYPOTHESIS",
    ("ENV_FACTOR", "GENE"): "ENV_GENE_HYPOTHESIS",
    ("ENV_FACTOR", "PROCESS"): "ENV_PROCESS_HYPOTHESIS",
    ("SPECIES", "TOXIN"): "SPECIES_TOXIN_HYPOTHESIS",
    ("SPECIES", "PROCESS"): "SPECIES_PROCESS_HYPOTHESIS",
    ("PROCESS", "TOXIN"): "PROCESS_TOXIN_HYPOTHESIS",
}


def canonical_pair_type(type1: str, type2: str):
    pair = tuple(sorted([type1, type2]))

    for allowed_pair, hyp_type in ALLOWED_PAIR_TYPES.items():
        if tuple(sorted(allowed_pair)) == pair:
            return hyp_type

    return None


def build_graph(edge_file: str) -> nx.Graph:
    edges = pd.read_csv(edge_file, encoding="utf-8-sig")

    G = nx.Graph()

    for _, row in edges.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])

        G.add_node(source, entity_type=str(row["Source_Type"]))
        G.add_node(target, entity_type=str(row["Target_Type"]))

        G.add_edge(
            source,
            target,
            relation=str(row["Relation"]),
            weight=float(row["Weight"]),
            paper_count=int(row["Paper_Count"]),
        )

    return G


def weighted_common_neighbor_score(
    G: nx.Graph,
    source: str,
    target: str,
    common_neighbors,
) -> float:
    score = 0.0

    for n in common_neighbors:
        w1 = G[source][n].get("weight", 1.0)
        w2 = G[target][n].get("weight", 1.0)
        score += min(w1, w2)

    return score


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = build_graph(EDGE_FILE)

    rows = []
    nodes = list(G.nodes())

    for source, target in combinations(nodes, 2):
        # We only want missing links
        if G.has_edge(source, target):
            continue

        source_type = G.nodes[source].get("entity_type", "")
        target_type = G.nodes[target].get("entity_type", "")

        hypothesis_type = canonical_pair_type(source_type, target_type)

        if hypothesis_type is None:
            continue

        common_neighbors = sorted(list(nx.common_neighbors(G, source, target)))

        if not common_neighbors:
            continue

        cn_count = len(common_neighbors)
        weighted_cn = weighted_common_neighbor_score(
            G, source, target, common_neighbors
        )

        source_degree = G.degree(source)
        target_degree = G.degree(target)
        source_weighted_degree = G.degree(source, weight="weight")
        target_weighted_degree = G.degree(target, weight="weight")

        # Adamic-Adar score
        aa_score = 0.0
        for n in common_neighbors:
            deg = G.degree(n)
            if deg > 1:
                aa_score += 1 / math.log(deg)

        # Preferential attachment
        pref_attach = source_degree * target_degree

        # Simple structural score
        structural_score = (
            cn_count * 2.0
            + weighted_cn * 0.5
            + aa_score * 2.0
            + pref_attach * 0.01
        )

        rows.append(
            {
                "Source": source,
                "Target": target,
                "Source_Type": source_type,
                "Target_Type": target_type,
                "Hypothesis_Type": hypothesis_type,
                "Common_Neighbor_Count": cn_count,
                "Weighted_Common_Neighbor_Score": round(weighted_cn, 4),
                "Adamic_Adar_Score": round(aa_score, 4),
                "Preferential_Attachment": pref_attach,
                "Source_Degree": source_degree,
                "Target_Degree": target_degree,
                "Source_Weighted_Degree": round(source_weighted_degree, 4),
                "Target_Weighted_Degree": round(target_weighted_degree, 4),
                "Structural_Score": round(structural_score, 4),
                "Bridge_Nodes": "; ".join(common_neighbors),
            }
        )

    hyp_df = pd.DataFrame(rows)

    if hyp_df.empty:
        print("No hypotheses generated.")
        return

    hyp_df = hyp_df.sort_values("Structural_Score", ascending=False)

    output_path = output_dir / OUTPUT_FILE
    hyp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved hypotheses: {output_path}")
    print(f"Total hypotheses generated: {len(hyp_df)}")
    print()
    print("Top hypotheses:")

    preview_cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Common_Neighbor_Count",
        "Structural_Score",
        "Bridge_Nodes",
    ]

    print(hyp_df[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
