#!/usr/bin/env python3
"""
Create a readable post-2015 dinoflagellate semantic knowledge graph.

Main improvements:
1. Aggregates duplicate source-target pairs.
2. Filters weak edges using minimum weight and/or top edge fraction.
3. Optionally retains only the strongest neighbors per node.
4. Preserves all nodes, including isolated nodes after filtering.
5. Uses Louvain communities for layout when available.
6. Scales node size by weighted degree.
7. Scales edge width and transparency by edge weight.
8. Labels all nodes.
9. Saves PNG, PDF, SVG, filtered edge CSV, and node statistics CSV.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT_CANDIDATES = [
    BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv",
    BASE / "FINAL_WORKSPACE/kg/post2015_dino_semantic_edges.csv",
    BASE / "FINAL_WORKSPACE/kg/dino_post_2015_semantic_edges.csv",
    BASE / "FINAL_WORKSPACE/figure_ready/dino_post2015_semantic_edges.csv",
]

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_post2015_dino_KG_filtered.png"
OUT_PDF = OUT_DIR / "Figure_post2015_dino_KG_filtered.pdf"
OUT_SVG = OUT_DIR / "Figure_post2015_dino_KG_filtered.svg"

OUT_FILTERED_EDGES = OUT_DIR / "Figure_post2015_dino_KG_filtered_edges.csv"
OUT_NODE_STATS = OUT_DIR / "Figure_post2015_dino_KG_node_statistics.csv"


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

FIGURE_TITLE = "Post-2015 Dinoflagellate Semantic Knowledge Graph"

# Filtering strategy:
# First retain edges meeting MIN_EDGE_WEIGHT.
# Then optionally retain only the strongest TOP_EDGE_FRACTION.
# Finally optionally keep TOP_NEIGHBORS_PER_NODE strongest edges per node.

MIN_EDGE_WEIGHT = 2

# Set to 1.0 to retain all edges passing MIN_EDGE_WEIGHT.
# Recommended values: 0.20–0.40 for dense networks.
TOP_EDGE_FRACTION = 0.30

# Retain up to this many strongest edges incident to each node.
# Set to None to disable.
TOP_NEIGHBORS_PER_NODE = 8

# Keep every original node in the plot even if it becomes isolated after filtering.
KEEP_ALL_NODES = True

# Community layout settings
LAYOUT_SEED = 42
SPRING_K_MULTIPLIER = 2.2
SPRING_ITERATIONS = 800

# Node sizes
MIN_NODE_SIZE = 380
MAX_NODE_SIZE = 3200

# Edge appearance
MIN_EDGE_WIDTH = 0.25
MAX_EDGE_WIDTH = 2.8
MIN_EDGE_ALPHA = 0.05
MAX_EDGE_ALPHA = 0.45

# Label appearance
LABEL_FONT_SIZE_MIN = 7
LABEL_FONT_SIZE_MAX = 13
LABEL_WRAP_WIDTH = 18

# Output quality
PNG_DPI = 600
FIGSIZE = (18, 14)


# ============================================================
# ENTITY TYPE COLORS
# ============================================================

ENTITY_COLORS = {
    "TOXIN": "#F57C00",
    "SXT_GENE": "#1F77B4",
    "DINO_TAXON": "#1B9E77",
    "CYANO_TAXON": "#56B4C2",
    "ENV_FACTOR": "#E6AB02",
    "BIOLOGICAL_PROCESS": "#CC79A7",
    "DETECTION_METHOD": "#A7A9AC",
    "OTHER": "#C7C7C7",
}

ENTITY_LABELS = {
    "TOXIN": "Toxin",
    "SXT_GENE": "sxt gene",
    "DINO_TAXON": "Dinoflagellate taxon",
    "CYANO_TAXON": "Cyanobacteria taxon",
    "ENV_FACTOR": "Environmental factor",
    "BIOLOGICAL_PROCESS": "Biological process",
    "DETECTION_METHOD": "Detection method",
    "OTHER": "Other",
}


# ============================================================
# HELPERS
# ============================================================

def find_input_file() -> Path:
    """Return the first existing candidate input file."""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate the post-2015 dinoflagellate edge file.\n"
        f"Searched:\n{searched}\n\n"
        "Edit INPUT_CANDIDATES near the top of the script."
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:
    """Find a column using case-insensitive matching."""
    lower_map = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise ValueError(
            f"None of the expected columns were found: {list(candidates)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


def clean_entity(value: object) -> str:
    """Clean an entity label without changing biological capitalization unnecessarily."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_entity_type(value: object) -> str:
    """Normalize entity type names to the project's standard ontology."""
    if pd.isna(value):
        return "OTHER"

    text = str(value).strip().upper()
    text = text.replace("-", "_").replace(" ", "_")

    mapping = {
        "TOXIN": "TOXIN",
        "TOXINS": "TOXIN",

        "SXT_GENE": "SXT_GENE",
        "SXT_GENES": "SXT_GENE",
        "GENE": "SXT_GENE",

        "DINO_TAXON": "DINO_TAXON",
        "DINO": "DINO_TAXON",
        "DINOFLAGELLATE": "DINO_TAXON",
        "DINOFLAGELLATE_TAXON": "DINO_TAXON",

        "CYANO_TAXON": "CYANO_TAXON",
        "CYANO": "CYANO_TAXON",
        "CYANOBACTERIA": "CYANO_TAXON",
        "CYANOBACTERIAL_TAXON": "CYANO_TAXON",

        "ENV_FACTOR": "ENV_FACTOR",
        "ENVIRONMENTAL_FACTOR": "ENV_FACTOR",
        "ENVIRONMENT": "ENV_FACTOR",

        "BIOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
        "BIO_PROCESS": "BIOLOGICAL_PROCESS",
        "PROCESS": "BIOLOGICAL_PROCESS",

        "DETECTION_METHOD": "DETECTION_METHOD",
        "METHOD": "DETECTION_METHOD",
        "ANALYTICAL_METHOD": "DETECTION_METHOD",
    }

    return mapping.get(text, "OTHER")


def infer_type_from_entity(entity: str) -> str:
    """
    Fallback ontology inference when no source/target type columns exist.
    This is only used when necessary.
    """
    text = entity.lower().strip()

    sxt_pattern = re.compile(r"\bsxt[a-z0-9/]*\b", re.IGNORECASE)
    if sxt_pattern.search(text) or "sxt gene" in text:
        return "SXT_GENE"

    toxins = [
        "saxitoxin",
        "neosaxitoxin",
        "neostx",
        "gonyautoxin",
        "gtx",
        "paralytic shellfish toxin",
        "paralytic shellfish toxins",
        "pst",
        "dcstx",
        "c1",
        "c2",
    ]
    if any(term in text for term in toxins):
        return "TOXIN"

    dino_taxa = [
        "alexandrium",
        "gymnodinium",
        "pyrodinium",
        "gonyaulax",
        "dinophysis",
        "prorocentrum",
        "karenia",
        "ostreopsis",
        "coolia",
        "centrodinium",
        "dinoflagellate",
    ]
    if any(term in text for term in dino_taxa):
        return "DINO_TAXON"

    cyano_taxa = [
        "cyanobacteria",
        "cyanobacterium",
        "aphanizomenon",
        "anabaena",
        "dolichospermum",
        "raphidiopsis",
        "cylindrospermopsis",
        "microseira",
        "lyngbya",
    ]
    if any(term in text for term in cyano_taxa):
        return "CYANO_TAXON"

    environmental_terms = [
        "temperature",
        "warming",
        "salinity",
        "light",
        "irradiance",
        "nitrate",
        "nitrogen",
        "phosphate",
        "phosphorus",
        "nutrient",
        "nutrients",
        "climate",
        "ph",
    ]
    if any(term == text or term in text for term in environmental_terms):
        return "ENV_FACTOR"

    detection_terms = [
        "hplc",
        "mass spectrometry",
        "lc-ms",
        "lc–ms",
        "mouse bioassay",
        "bioassay",
        "chromatography",
        "elisa",
        "pcr",
    ]
    if any(term in text for term in detection_terms):
        return "DETECTION_METHOD"

    return "BIOLOGICAL_PROCESS"


def wrap_label(text: str, width: int = LABEL_WRAP_WIDTH) -> str:
    """Wrap labels without splitting words."""
    words = str(text).split()
    if not words:
        return ""

    lines: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        projected = current_length + len(word) + (1 if current else 0)

        if projected <= width:
            current.append(word)
            current_length = projected
        else:
            lines.append(" ".join(current))
            current = [word]
            current_length = len(word)

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def scale_values(
    values: dict[str, float],
    minimum: float,
    maximum: float,
    log_transform: bool = False,
) -> dict[str, float]:
    """Scale dictionary values into a fixed range."""
    if not values:
        return {}

    keys = list(values.keys())
    arr = np.array([float(values[key]) for key in keys], dtype=float)

    if log_transform:
        arr = np.log1p(arr)

    low = float(arr.min())
    high = float(arr.max())

    if math.isclose(low, high):
        midpoint = (minimum + maximum) / 2
        return {key: midpoint for key in keys}

    scaled = minimum + (arr - low) * (maximum - minimum) / (high - low)
    return dict(zip(keys, scaled))


# ============================================================
# LOAD AND STANDARDIZE EDGES
# ============================================================

def load_edges(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the input edge table and return:
    1. Aggregated edge dataframe
    2. Node ontology dataframe
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file contains no rows: {path}")

    source_col = first_existing_column(
        df,
        ["source", "Source", "node1", "entity1", "from", "u"],
    )
    target_col = first_existing_column(
        df,
        ["target", "Target", "node2", "entity2", "to", "v"],
    )

    weight_col = first_existing_column(
        df,
        [
            "weight",
            "Weight",
            "edge_weight",
            "cooccurrence_count",
            "co_occurrence_count",
            "count",
            "frequency",
            "n",
        ],
        required=False,
    )

    source_type_col = first_existing_column(
        df,
        [
            "source_type",
            "Source_Type",
            "entity1_type",
            "node1_type",
            "type_source",
        ],
        required=False,
    )

    target_type_col = first_existing_column(
        df,
        [
            "target_type",
            "Target_Type",
            "entity2_type",
            "node2_type",
            "type_target",
        ],
        required=False,
    )

    edge_df = pd.DataFrame(
        {
            "source": df[source_col].map(clean_entity),
            "target": df[target_col].map(clean_entity),
        }
    )

    if weight_col is None:
        edge_df["weight"] = 1.0
    else:
        edge_df["weight"] = pd.to_numeric(
            df[weight_col],
            errors="coerce",
        ).fillna(1.0)

    if source_type_col:
        edge_df["source_type"] = df[source_type_col].map(normalize_entity_type)
    else:
        edge_df["source_type"] = edge_df["source"].map(infer_type_from_entity)

    if target_type_col:
        edge_df["target_type"] = df[target_type_col].map(normalize_entity_type)
    else:
        edge_df["target_type"] = edge_df["target"].map(infer_type_from_entity)

    # Remove invalid or self-loop edges
    edge_df = edge_df[
        (edge_df["source"] != "")
        & (edge_df["target"] != "")
        & (edge_df["source"] != edge_df["target"])
    ].copy()

    # Canonical undirected pair
    canonical_pairs = edge_df.apply(
        lambda row: tuple(sorted((row["source"], row["target"]))),
        axis=1,
    )

    edge_df["u"] = [pair[0] for pair in canonical_pairs]
    edge_df["v"] = [pair[1] for pair in canonical_pairs]

    # Build node type table before edge aggregation
    source_nodes = edge_df[["source", "source_type"]].rename(
        columns={"source": "node", "source_type": "entity_type"}
    )
    target_nodes = edge_df[["target", "target_type"]].rename(
        columns={"target": "node", "target_type": "entity_type"}
    )

    node_types = pd.concat([source_nodes, target_nodes], ignore_index=True)

    # If one node has multiple types, use the most frequent non-OTHER type
    node_types = (
        node_types.groupby("node")["entity_type"]
        .agg(
            lambda x: (
                x[x != "OTHER"].mode().iloc[0]
                if not x[x != "OTHER"].mode().empty
                else "OTHER"
            )
        )
        .reset_index()
    )

    # Aggregate duplicate undirected edges
    aggregated = (
        edge_df.groupby(["u", "v"], as_index=False)
        .agg(weight=("weight", "sum"))
        .rename(columns={"u": "source", "v": "target"})
    )

    aggregated["weight"] = pd.to_numeric(
        aggregated["weight"],
        errors="coerce",
    ).fillna(1.0)

    return aggregated, node_types


# ============================================================
# EDGE FILTERING
# ============================================================

def filter_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Apply minimum-weight, global top-fraction, and local top-neighbor filters."""
    if edges.empty:
        return edges.copy()

    filtered = edges.copy()

    # 1. Minimum edge weight
    filtered = filtered[filtered["weight"] >= MIN_EDGE_WEIGHT].copy()

    if filtered.empty:
        raise ValueError(
            f"No edges remain after MIN_EDGE_WEIGHT={MIN_EDGE_WEIGHT}. "
            "Lower MIN_EDGE_WEIGHT to 1."
        )

    # 2. Keep strongest global fraction
    if 0 < TOP_EDGE_FRACTION < 1:
        keep_n = max(1, math.ceil(len(filtered) * TOP_EDGE_FRACTION))
        filtered = filtered.nlargest(keep_n, "weight").copy()

    # 3. Keep strongest neighbors for each node
    if TOP_NEIGHBORS_PER_NODE is not None and TOP_NEIGHBORS_PER_NODE > 0:
        incident_rows: list[pd.DataFrame] = []

        all_nodes = pd.unique(
            pd.concat([filtered["source"], filtered["target"]], ignore_index=True)
        )

        for node in all_nodes:
            incident = filtered[
                (filtered["source"] == node) | (filtered["target"] == node)
            ]
            incident_rows.append(
                incident.nlargest(TOP_NEIGHBORS_PER_NODE, "weight")
            )

        filtered = (
            pd.concat(incident_rows, ignore_index=True)
            .drop_duplicates(subset=["source", "target"])
            .copy()
        )

    return filtered.sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
    )


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    filtered_edges: pd.DataFrame,
    node_types: pd.DataFrame,
) -> nx.Graph:
    graph = nx.Graph()

    type_lookup = dict(
        zip(node_types["node"], node_types["entity_type"])
    )

    if KEEP_ALL_NODES:
        for node, entity_type in type_lookup.items():
            graph.add_node(node, entity_type=entity_type)

    for row in filtered_edges.itertuples(index=False):
        source = row.source
        target = row.target
        weight = float(row.weight)

        graph.add_node(
            source,
            entity_type=type_lookup.get(source, infer_type_from_entity(source)),
        )
        graph.add_node(
            target,
            entity_type=type_lookup.get(target, infer_type_from_entity(target)),
        )
        graph.add_edge(source, target, weight=weight)

    return graph


# ============================================================
# COMMUNITY-AWARE LAYOUT
# ============================================================

def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Detect Louvain communities using NetworkX, with a greedy fallback."""
    graph_for_community = graph.copy()
    graph_for_community.remove_nodes_from(
        list(nx.isolates(graph_for_community))
    )

    partition: dict[str, int] = {}

    if graph_for_community.number_of_nodes() == 0:
        return {node: index for index, node in enumerate(graph.nodes())}

    try:
        communities = nx.community.louvain_communities(
            graph_for_community,
            weight="weight",
            seed=LAYOUT_SEED,
        )
    except (AttributeError, ImportError):
        communities = nx.community.greedy_modularity_communities(
            graph_for_community,
            weight="weight",
        )

    for community_index, community_nodes in enumerate(communities):
        for node in community_nodes:
            partition[node] = community_index

    next_index = len(communities)
    for node in graph.nodes():
        if node not in partition:
            partition[node] = next_index
            next_index += 1

    return partition


def community_layout(
    graph: nx.Graph,
    partition: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Arrange communities around a circle and calculate a local spring layout
    within each community.
    """
    community_to_nodes: dict[int, list[str]] = {}

    for node, community in partition.items():
        community_to_nodes.setdefault(community, []).append(node)

    community_ids = sorted(community_to_nodes)
    n_communities = len(community_ids)

    if n_communities <= 1:
        n_nodes = max(graph.number_of_nodes(), 1)
        return nx.spring_layout(
            graph,
            seed=LAYOUT_SEED,
            weight="weight",
            k=SPRING_K_MULTIPLIER / math.sqrt(n_nodes),
            iterations=SPRING_ITERATIONS,
        )

    positions: dict[str, np.ndarray] = {}

    circle_radius = 4.6
    angles = np.linspace(0, 2 * np.pi, n_communities, endpoint=False)

    for position_index, community_id in enumerate(community_ids):
        nodes = community_to_nodes[community_id]

        center = np.array(
            [
                circle_radius * np.cos(angles[position_index]),
                circle_radius * np.sin(angles[position_index]),
            ]
        )

        subgraph = graph.subgraph(nodes).copy()

        if len(nodes) == 1:
            local_positions = {nodes[0]: np.array([0.0, 0.0])}
        else:
            local_positions = nx.spring_layout(
                subgraph,
                seed=LAYOUT_SEED + position_index,
                weight="weight",
                k=1.6 / math.sqrt(len(nodes)),
                iterations=500,
                scale=1.5,
            )

        local_scale = min(2.0, 0.65 + 0.12 * math.sqrt(len(nodes)))

        for node, local_xy in local_positions.items():
            positions[node] = center + np.asarray(local_xy) * local_scale

    # Light global relaxation while keeping broad community structure
    positions = nx.spring_layout(
        graph,
        pos=positions,
        fixed=None,
        seed=LAYOUT_SEED,
        weight="weight",
        k=SPRING_K_MULTIPLIER / math.sqrt(max(graph.number_of_nodes(), 1)),
        iterations=120,
    )

    return positions


# ============================================================
# PLOT
# ============================================================

def plot_graph(
    graph: nx.Graph,
    filtered_edges: pd.DataFrame,
    node_types: pd.DataFrame,
) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph contains no nodes.")

    partition = detect_communities(graph)
    positions = community_layout(graph, partition)

    weighted_degree = dict(graph.degree(weight="weight"))
    unweighted_degree = dict(graph.degree())

    node_sizes = scale_values(
        weighted_degree,
        MIN_NODE_SIZE,
        MAX_NODE_SIZE,
        log_transform=True,
    )

    label_sizes = scale_values(
        weighted_degree,
        LABEL_FONT_SIZE_MIN,
        LABEL_FONT_SIZE_MAX,
        log_transform=True,
    )

    edge_weights = {
        f"{u}||{v}": data.get("weight", 1.0)
        for u, v, data in graph.edges(data=True)
    }

    edge_widths = scale_values(
        edge_weights,
        MIN_EDGE_WIDTH,
        MAX_EDGE_WIDTH,
        log_transform=True,
    )

    edge_alphas = scale_values(
        edge_weights,
        MIN_EDGE_ALPHA,
        MAX_EDGE_ALPHA,
        log_transform=True,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("white")

    # Draw edges individually to support weight-specific alpha
    for u, v, data in graph.edges(data=True):
        key = f"{u}||{v}"
        weight = float(data.get("weight", 1.0))

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(u, v)],
            width=edge_widths[key],
            alpha=edge_alphas[key],
            edge_color="#6E6E6E",
            ax=ax,
        )

    # Draw nodes grouped by ontology type
    present_types = []

    for entity_type in ENTITY_COLORS:
        nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("entity_type", "OTHER") == entity_type
        ]

        if not nodes:
            continue

        present_types.append(entity_type)

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_size=[node_sizes[node] for node in nodes],
            node_color=ENTITY_COLORS[entity_type],
            edgecolors="black",
            linewidths=0.8,
            alpha=0.94,
            ax=ax,
        )

    # Labels: all nodes
    for node, (x, y) in positions.items():
        font_size = float(label_sizes[node])

        ax.text(
            x,
            y,
            wrap_label(node),
            fontsize=font_size,
            fontfamily="DejaVu Sans",
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            zorder=10,
        )

    ax.set_title(
        FIGURE_TITLE,
        fontsize=24,
        fontweight="bold",
        pad=24,
    )

    # Legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ENTITY_COLORS[entity_type],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=10,
            label=ENTITY_LABELS[entity_type],
        )
        for entity_type in present_types
    ]

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            title="Entity type",
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            fontsize=11,
            title_fontsize=12,
            ncol=2,
        )
        legend.get_title().set_fontweight("bold")

    ax.axis("off")
    ax.margins(0.12)

    plt.tight_layout()

    fig.savefig(
        OUT_PNG,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT_SVG,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # Save node statistics
    type_lookup = dict(
        zip(node_types["node"], node_types["entity_type"])
    )

    stats = pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "entity_type": [
                graph.nodes[node].get(
                    "entity_type",
                    type_lookup.get(node, "OTHER"),
                )
                for node in graph.nodes()
            ],
            "degree": [
                int(unweighted_degree.get(node, 0))
                for node in graph.nodes()
            ],
            "weighted_degree": [
                float(weighted_degree.get(node, 0.0))
                for node in graph.nodes()
            ],
            "community": [
                int(partition.get(node, -1))
                for node in graph.nodes()
            ],
        }
    ).sort_values(
        ["weighted_degree", "degree"],
        ascending=False,
    )

    stats.to_csv(OUT_NODE_STATS, index=False)
    filtered_edges.to_csv(OUT_FILTERED_EDGES, index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    input_file = find_input_file()

    print("=" * 70)
    print("POST-2015 DINOFLAGELLATE KG: FILTERED NETWORK FIGURE")
    print("=" * 70)
    print(f"Input: {input_file}")

    all_edges, node_types = load_edges(input_file)

    print(f"Original unique nodes: {node_types['node'].nunique():,}")
    print(f"Original aggregated edges: {len(all_edges):,}")
    print(
        f"Original edge-weight range: "
        f"{all_edges['weight'].min():.2f}–{all_edges['weight'].max():.2f}"
    )

    filtered_edges = filter_edges(all_edges)

    print(f"\nFiltering settings:")
    print(f"  Minimum edge weight: {MIN_EDGE_WEIGHT}")
    print(f"  Top edge fraction: {TOP_EDGE_FRACTION}")
    print(f"  Top neighbors per node: {TOP_NEIGHBORS_PER_NODE}")

    print(f"\nFiltered edges retained: {len(filtered_edges):,}")
    print(
        f"Edges removed: "
        f"{len(all_edges) - len(filtered_edges):,} "
        f"({100 * (1 - len(filtered_edges) / len(all_edges)):.1f}%)"
    )

    graph = build_graph(filtered_edges, node_types)

    print(f"Plotted nodes: {graph.number_of_nodes():,}")
    print(f"Plotted edges: {graph.number_of_edges():,}")
    print(f"Connected components: {nx.number_connected_components(graph):,}")
    print(f"Isolated nodes: {len(list(nx.isolates(graph))):,}")

    plot_graph(graph, filtered_edges, node_types)

    print("\nSaved:")
    print(f"  PNG: {OUT_PNG}")
    print(f"  PDF: {OUT_PDF}")
    print(f"  SVG: {OUT_SVG}")
    print(f"  Filtered edges: {OUT_FILTERED_EDGES}")
    print(f"  Node statistics: {OUT_NODE_STATS}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
