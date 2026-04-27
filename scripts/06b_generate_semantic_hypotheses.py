from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pandas as pd
import networkx as nx


EDGE_FILE = "data/graphs_semantic/semantic_pre2016_edges_filtered.csv"
OUTPUT_DIR = "results/semantic_hypotheses"
OUTPUT_FILE = "STX_SEMANTIC_HYPOTHESES.csv"


ALLOWED_PAIR_TYPES = {
    ("GENE", "TOXIN"): "GENE_TOXIN_HYPOTHESIS",
    ("GENE", "TOXIN_PHENOTYPE"): "GENE_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("GENE", "PHRASE_TOXIN_PHENOTYPE"): "GENE_TOXIN_PHENOTYPE_HYPOTHESIS",

    ("GENE", "BIOLOGICAL_MECHANISM"): "GENE_MECHANISM_HYPOTHESIS",
    ("GENE", "REGULATION_EXPRESSION"): "GENE_REGULATION_HYPOTHESIS",
    ("GENE", "GENE_PRESENCE_ABSENCE"): "GENE_PRESENCE_ABSENCE_HYPOTHESIS",
    ("GENE", "EVOLUTIONARY_PROCESS"): "GENE_EVOLUTION_HYPOTHESIS",
    ("GENE", "GENE_DOMAIN"): "GENE_DOMAIN_HYPOTHESIS",

    ("GENE", "PHRASE_GENE_MECHANISM"): "GENE_MECHANISM_PHRASE_HYPOTHESIS",
    ("GENE", "PHRASE_REGULATION"): "GENE_REGULATION_PHRASE_HYPOTHESIS",
    ("GENE", "PHRASE_EVOLUTION"): "GENE_EVOLUTION_PHRASE_HYPOTHESIS",
    ("GENE", "PHRASE_BIOSYNTHESIS"): "GENE_BIOSYNTHESIS_PHRASE_HYPOTHESIS",

    ("ENV_FACTOR", "TOXIN"): "ENV_TOXIN_HYPOTHESIS",
    ("ENV_FACTOR", "TOXIN_PHENOTYPE"): "ENV_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("ENV_FACTOR", "PHRASE_TOXIN_PHENOTYPE"): "ENV_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("ENV_FACTOR", "BIOLOGICAL_MECHANISM"): "ENV_MECHANISM_HYPOTHESIS",
    ("ENV_FACTOR", "REGULATION_EXPRESSION"): "ENV_REGULATION_HYPOTHESIS",
    ("ENV_FACTOR", "GENE"): "ENV_GENE_HYPOTHESIS",
    ("ENV_FACTOR", "PHRASE_ENVIRONMENT"): "ENVIRONMENT_PHRASE_HYPOTHESIS",

    ("SPECIES", "GENE"): "SPECIES_GENE_HYPOTHESIS",
    ("SPECIES", "GENE_PRESENCE_ABSENCE"): "SPECIES_GENE_PRESENCE_ABSENCE_HYPOTHESIS",
    ("SPECIES", "REGULATION_EXPRESSION"): "SPECIES_REGULATION_HYPOTHESIS",
    ("SPECIES", "BIOLOGICAL_MECHANISM"): "SPECIES_MECHANISM_HYPOTHESIS",
    ("SPECIES", "EVOLUTIONARY_PROCESS"): "SPECIES_EVOLUTION_HYPOTHESIS",
    ("SPECIES", "TOXIN_PHENOTYPE"): "SPECIES_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("SPECIES", "PHRASE_TOXIN_PHENOTYPE"): "SPECIES_TOXIN_PHENOTYPE_HYPOTHESIS",

    ("EVOLUTIONARY_PROCESS", "GENE_PRESENCE_ABSENCE"): "EVOLUTION_GENE_PRESENCE_HYPOTHESIS",
    ("EVOLUTIONARY_PROCESS", "BIOLOGICAL_MECHANISM"): "EVOLUTION_MECHANISM_HYPOTHESIS",
    ("EVOLUTIONARY_PROCESS", "PHRASE_GENE_MECHANISM"): "EVOLUTION_GENE_MECHANISM_HYPOTHESIS",

    ("REGULATION_EXPRESSION", "TOXIN_PHENOTYPE"): "REGULATION_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("REGULATION_EXPRESSION", "PHRASE_TOXIN_PHENOTYPE"): "REGULATION_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("BIOLOGICAL_MECHANISM", "TOXIN_PHENOTYPE"): "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS",
    ("BIOLOGICAL_MECHANISM", "PHRASE_TOXIN_PHENOTYPE"): "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS",
}


LOW_VALUE_NODES = {
    "saxitoxin",
    "paralytic shellfish toxins",
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

        if source.lower() in LOW_VALUE_NODES and target.lower() in LOW_VALUE_NODES:
            continue

        G.add_node(source, entity_type=str(row["Source_Type"]))
        G.add_node(target, entity_type=str(row["Target_Type"]))

        G.add_edge(
            source,
            target,
            relation=str(row["Relation"]),
            relation_hints=str(row.get("Relation_Hints", "")),
            weight=float(row["Weight"]),
            paper_count=int(row["Paper_Count"]),
        )

    return G


def weighted_common_neighbor_score(G: nx.Graph, source: str, target: str, common_neighbors) -> float:
    score = 0.0

    for n in common_neighbors:
        w1 = G[source][n].get("weight", 1.0)
        w2 = G[target][n].get("weight", 1.0)
        score += min(w1, w2)

    return score


def collect_bridge_relation_hints(G: nx.Graph, source: str, target: str, common_neighbors) -> str:
    hints = []

    for n in common_neighbors:
        hints.extend(str(G[source][n].get("relation_hints", "")).split(";"))
        hints.extend(str(G[target][n].get("relation_hints", "")).split(";"))

    hints = sorted(set([h.strip() for h in hints if h.strip() and h.strip().lower() != "nan"]))
    return ";".join(hints)


def semantic_bonus(source_type: str, target_type: str, source: str, target: str) -> float:
    text = f"{source} {target}".lower()
    bonus = 0.0

    if "sxt" in text:
        bonus += 2.0
    if "gene expression" in text or "regulation" in text or "transcription" in text:
        bonus += 1.5
    if "biosynthesis" in text or "toxin production" in text:
        bonus += 1.5
    if "presence" in text or "absence" in text or "loss" in text:
        bonus += 1.5
    if "evolution" in text or "phylogen" in text or "divergence" in text or "hgt" in text:
        bonus += 1.5
    if "temperature" in text or "salinity" in text or "nitrogen" in text or "nitrate" in text:
        bonus += 1.0

    return bonus


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = build_graph(EDGE_FILE)

    print(f"Semantic graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    rows = []
    nodes = list(G.nodes())

    for source, target in combinations(nodes, 2):
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
        weighted_cn = weighted_common_neighbor_score(G, source, target, common_neighbors)

        source_degree = G.degree(source)
        target_degree = G.degree(target)
        source_weighted_degree = G.degree(source, weight="weight")
        target_weighted_degree = G.degree(target, weight="weight")

        aa_score = 0.0
        for n in common_neighbors:
            deg = G.degree(n)
            if deg > 1:
                aa_score += 1 / math.log(deg)

        pref_attach = source_degree * target_degree
        bonus = semantic_bonus(source_type, target_type, source, target)
        relation_hints = collect_bridge_relation_hints(G, source, target, common_neighbors)

        structural_score = (
            cn_count * 2.0
            + weighted_cn * 0.5
            + aa_score * 2.0
            + pref_attach * 0.005
            + bonus
        )

        rows.append({
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
            "Semantic_Bonus": bonus,
            "Structural_Score": round(structural_score, 4),
            "Bridge_Nodes": "; ".join(common_neighbors),
            "Bridge_Relation_Hints": relation_hints,
        })

    hyp_df = pd.DataFrame(rows)

    if hyp_df.empty:
        print("No semantic hypotheses generated.")
        return

    hyp_df = hyp_df.sort_values("Structural_Score", ascending=False)

    output_path = output_dir / OUTPUT_FILE
    hyp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved semantic hypotheses: {output_path}")
    print(f"Total semantic hypotheses generated: {len(hyp_df)}")
    print()
    print("Top semantic hypotheses:")
    preview_cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Common_Neighbor_Count",
        "Structural_Score",
        "Semantic_Bonus",
        "Bridge_Nodes",
        "Bridge_Relation_Hints",
    ]
    print(hyp_df[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
