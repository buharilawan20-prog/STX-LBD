from __future__ import annotations

from pathlib import Path
import pandas as pd
import networkx as nx


TRAIN_EDGE_FILE = "data/graphs/dino_pre2016_edges.csv"
TEST_EDGE_FILE = "data/graphs/dino_post2015_edges.csv"

OUTPUT_DIR = "data/graphs"

TRAIN_NODE_FILE = "dino_pre2016_nodes.csv"
TEST_NODE_FILE = "dino_post2015_nodes.csv"
TRAIN_GRAPHML_FILE = "dino_pre2016.graphml"
TEST_GRAPHML_FILE = "dino_post2015.graphml"
SUMMARY_FILE = "graph_summary.csv"


def build_graph(edge_file: str) -> nx.Graph:
    df = pd.read_csv(edge_file, encoding="utf-8-sig")

    G = nx.Graph()

    for _, row in df.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])

        source_type = str(row["Source_Type"])
        target_type = str(row["Target_Type"])
        relation = str(row["Relation"])
        weight = float(row["Weight"])
        paper_count = int(row["Paper_Count"])

        G.add_node(source, entity_type=source_type)
        G.add_node(target, entity_type=target_type)

        G.add_edge(
            source,
            target,
            relation=relation,
            weight=weight,
            paper_count=paper_count,
        )

    return G


def make_node_table(G: nx.Graph) -> pd.DataFrame:
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    if G.number_of_nodes() > 1:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    else:
        betweenness = {node: 0 for node in G.nodes()}

    rows = []

    for node, attrs in G.nodes(data=True):
        rows.append({
            "Node": node,
            "Entity_Type": attrs.get("entity_type", ""),
            "Degree": degree.get(node, 0),
            "Weighted_Degree": weighted_degree.get(node, 0),
            "Betweenness": betweenness.get(node, 0),
        })

    node_df = pd.DataFrame(rows)
    node_df = node_df.sort_values(
        ["Weighted_Degree", "Degree"],
        ascending=False
    )

    return node_df


def graph_summary(G: nx.Graph, name: str) -> dict:
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        components = nx.number_connected_components(G)
        largest_component = len(max(nx.connected_components(G), key=len))
    else:
        density = 0
        components = 0
        largest_component = 0

    return {
        "Graph": name,
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": density,
        "Connected_Components": components,
        "Largest_Component_Size": largest_component,
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    G_train = build_graph(TRAIN_EDGE_FILE)
    G_test = build_graph(TEST_EDGE_FILE)

    train_nodes = make_node_table(G_train)
    test_nodes = make_node_table(G_test)

    train_nodes.to_csv(output_dir / TRAIN_NODE_FILE, index=False, encoding="utf-8-sig")
    test_nodes.to_csv(output_dir / TEST_NODE_FILE, index=False, encoding="utf-8-sig")

    nx.write_graphml(G_train, output_dir / TRAIN_GRAPHML_FILE)
    nx.write_graphml(G_test, output_dir / TEST_GRAPHML_FILE)

    summary = pd.DataFrame([
        graph_summary(G_train, "dino_pre2016_train"),
        graph_summary(G_test, "dino_post2015_test"),
    ])

    summary.to_csv(output_dir / SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(f"  - {output_dir / TRAIN_NODE_FILE}")
    print(f"  - {output_dir / TEST_NODE_FILE}")
    print(f"  - {output_dir / TRAIN_GRAPHML_FILE}")
    print(f"  - {output_dir / TEST_GRAPHML_FILE}")
    print(f"  - {output_dir / SUMMARY_FILE}")
    print()
    print("Graph summary:")
    print(summary.to_string(index=False))
    print()
    print("Top train nodes:")
    print(train_nodes.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
