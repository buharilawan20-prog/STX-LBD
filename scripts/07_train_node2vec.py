from __future__ import annotations

from pathlib import Path
import pandas as pd
import networkx as nx


EDGE_FILE = "data/graphs/dino_pre2016_edges.csv"
OUTPUT_DIR = "results/embeddings"
OUTPUT_FILE = "STX_NODE_EMBEDDINGS.csv"


DIMENSIONS = 64
WALK_LENGTH = 20
NUM_WALKS = 100
WINDOW = 5
MIN_COUNT = 1
BATCH_WORDS = 4
WORKERS = 2
P = 1
Q = 1


def build_graph(edge_file: str) -> nx.Graph:
    df = pd.read_csv(edge_file, encoding="utf-8-sig")

    G = nx.Graph()

    for _, row in df.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])
        weight = float(row["Weight"])

        G.add_edge(source, target, weight=weight)

    return G


def main():
    try:
        from node2vec import Node2Vec
    except ImportError:
        raise ImportError(
            "node2vec is not installed. Run:\n"
            "pip install node2vec gensim scipy joblib"
        )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = build_graph(EDGE_FILE)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Training Node2Vec...")

    node2vec = Node2Vec(
        G,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=WORKERS,
        p=P,
        q=Q,
        weight_key="weight",
        quiet=False,
    )

    model = node2vec.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS,
    )

    rows = []

    for node in G.nodes():
        vector = model.wv[str(node)]
        row = {"Node": node}

        for i, value in enumerate(vector):
            row[f"emb_{i}"] = float(value)

        rows.append(row)

    emb_df = pd.DataFrame(rows)

    output_path = output_dir / OUTPUT_FILE
    emb_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved embeddings: {output_path}")
    print(f"Total embedded nodes: {len(emb_df)}")


if __name__ == "__main__":
    main()
