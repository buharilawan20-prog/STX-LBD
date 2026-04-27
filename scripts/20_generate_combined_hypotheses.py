from pathlib import Path
import pandas as pd
import networkx as nx
from itertools import combinations


INPUT_FILE = "data/graphs_transfer/combined_cyano_dino_train_edges.csv"
OUTPUT_FILE = "results/combined/STX_COMBINED_HYPOTHESES.csv"


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    hypotheses = []

    nodes = list(G.nodes())

    for u, v in combinations(nodes, 2):
        if G.has_edge(u, v):
            continue

        common = list(nx.common_neighbors(G, u, v))

        if not common:
            continue

        score = 0
        for w in common:
            deg = len(list(G.neighbors(w)))
            if deg > 1:
                import math
                score += 1 / math.log(deg)

        hypotheses.append({
            "Source": u,
            "Target": v,
            "Score": score,
            "Common_Neighbors": len(common),
            "Bridge_Nodes": ";".join(common[:20])
        })

    hyp_df = pd.DataFrame(hypotheses)
    hyp_df = hyp_df.sort_values("Score", ascending=False)

    Path("results/combined").mkdir(parents=True, exist_ok=True)
    hyp_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved combined hypotheses: {OUTPUT_FILE}")
    print(f"Total hypotheses: {len(hyp_df)}")


if __name__ == "__main__":
    main()
