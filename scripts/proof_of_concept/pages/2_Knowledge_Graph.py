
from pathlib import Path
import json
import sys
import tempfile

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from query_engine import STXLBD
from ui import apply_global_style, hero, render_sidebar, section


apply_global_style()
render_sidebar()

def locate_graph_file():
    workspace = PROJECT_ROOT / "FINAL_WORKSPACE"
    preferred = [
        workspace / "knowledge_graphs" / "dino_all_semantic_edges.csv",
        workspace / "kg" / "dino_all_semantic_edges.csv",
        workspace / "graphs" / "dino_all_semantic_edges.csv",
        workspace / "dino_all_semantic_edges.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    if workspace.exists():
        exact = list(workspace.rglob("dino_all_semantic_edges.csv"))
        if exact:
            return sorted(exact, key=lambda p: len(p.parts))[0]

        fallback = list(workspace.rglob("*dino*all*semantic*edge*.csv"))
        if fallback:
            return sorted(fallback, key=lambda p: len(p.parts))[0]

    return None

@st.cache_data
def load_graph_dataframe(path_str):
    frame = pd.read_csv(path_str)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame

@st.cache_resource
def load_query_engine():
    try:
        return STXLBD()
    except Exception:
        return None

graph_path = locate_graph_file()
if graph_path is None:
    st.error(
        "The final dinoflagellate edge list was not found. "
        "Expected filename: `dino_all_semantic_edges.csv` under `FINAL_WORKSPACE`."
    )
    st.stop()

df = load_graph_dataframe(str(graph_path))
engine = load_query_engine()

def find_col(*candidates):
    lower = {str(c).casefold(): c for c in df.columns}
    return next((lower[c.casefold()] for c in candidates if c.casefold() in lower), None)

source_col = find_col("Source", "Entity_1", "Node_1")
target_col = find_col("Target", "Entity_2", "Node_2")
weight_col = find_col("Weight", "Count", "Support")
source_type_col = find_col("Source_Type", "Entity_1_Type")
target_type_col = find_col("Target_Type", "Entity_2_Type")

if source_col is None or target_col is None:
    st.error("The graph file must contain source and target columns.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

df = df.dropna(subset=[source_col, target_col]).copy()
df[source_col] = df[source_col].astype(str).str.strip()
df[target_col] = df[target_col].astype(str).str.strip()

if weight_col:
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
else:
    weight_col = "_weight"
    df[weight_col] = 1.0

def node_type_lookup():
    lookup = {}
    if source_type_col:
        for node, node_type in zip(df[source_col], df[source_type_col]):
            if pd.notna(node_type):
                lookup[str(node)] = str(node_type).strip()
    if target_type_col:
        for node, node_type in zip(df[target_col], df[target_type_col]):
            if pd.notna(node_type):
                lookup[str(node)] = str(node_type).strip()
    return lookup

types = node_type_lookup()
all_nodes = sorted(set(df[source_col]) | set(df[target_col]))

TYPE_COLORS = {
    "SXT_GENE": "#287DB2",
    "TOXIN": "#E76F00",
    "DINO_TAXON": "#159A73",
    "CYANO_TAXON": "#58AFCC",
    "ENV_FACTOR": "#F2A900",
    "BIOLOGICAL_PROCESS": "#CF6EA3",
    "DETECTION_METHOD": "#8C8C8C",
    "UNKNOWN": "#6B7280",
}

hero(
    "🕸️ STX-LBD Knowledge Graph Explorer",
    "Explore entity-centered semantic neighborhoods, inspect connected biological "
    "relationships, and retrieve linked STX-LBD hypotheses.",
)

tab_graph, tab_node, tab_hypotheses = st.tabs(
    ["Interactive network", "Node profile", "Connected hypotheses"]
)

with tab_graph:
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        focus_node = st.selectbox(
            "Focus entity",
            options=all_nodes,
            placeholder="Type or select an entity...",
        )

    with c2:
        depth = st.selectbox(
            "Neighborhood depth",
            options=[1, 2],
            index=0,
            help="Depth 1 displays direct neighbors. Depth 2 also includes neighbors of neighbors.",
        )

    with c3:
        max_edges = st.slider(
            "Maximum displayed edges",
            min_value=25,
            max_value=750,
            value=150,
            step=25,
        )

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        available_types = sorted({types.get(node, "UNKNOWN") for node in all_nodes})
        selected_types = st.multiselect(
            "Entity types",
            options=available_types,
            default=available_types,
        )

    with filter2:
        min_weight = float(df[weight_col].min())
        max_weight = float(df[weight_col].max())
        minimum_weight = st.slider(
            "Minimum semantic support",
            min_value=min_weight,
            max_value=max_weight,
            value=min_weight,
            step=1.0 if max_weight - min_weight >= 1 else 0.1,
        )

    with filter3:
        physics_enabled = st.checkbox(
            "Enable network physics",
            value=True,
            help="Turn this off after the layout stabilizes.",
        )

    full_graph = nx.Graph()
    for _, row in df.iterrows():
        full_graph.add_edge(
            row[source_col],
            row[target_col],
            weight=float(row[weight_col]),
        )

    selected_nodes = {focus_node}
    frontier = {focus_node}

    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            if node in full_graph:
                next_frontier.update(full_graph.neighbors(node))
        selected_nodes.update(next_frontier)
        frontier = next_frontier

    selected_nodes = {
        node for node in selected_nodes
        if node == focus_node or types.get(node, "UNKNOWN") in selected_types
    }

    filtered = df[
        df[source_col].isin(selected_nodes)
        & df[target_col].isin(selected_nodes)
        & (df[weight_col] >= minimum_weight)
    ].copy()

    filtered = filtered.sort_values(weight_col, ascending=False).head(max_edges)

    if filtered.empty:
        st.info("No relationships remain after applying the selected filters.")
        st.stop()

    display_graph = nx.Graph()
    for _, row in filtered.iterrows():
        display_graph.add_edge(
            row[source_col],
            row[target_col],
            weight=float(row[weight_col]),
        )

    net = Network(
        height="720px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1F2937",
        notebook=False,
        cdn_resources="in_line",
    )

    for node in display_graph.nodes():
        node_type = types.get(node, "UNKNOWN")
        degree = display_graph.degree(node)
        net.add_node(
            node,
            label=node,
            title=f"{node}<br>Type: {node_type}<br>Displayed degree: {degree}",
            color=TYPE_COLORS.get(node_type, TYPE_COLORS["UNKNOWN"]),
            size=32 if node == focus_node else 14 + min(degree, 12),
            borderWidth=5 if node == focus_node else 1,
        )

    for source, target, edge_data in display_graph.edges(data=True):
        weight = float(edge_data.get("weight", 1.0))
        net.add_edge(
            source,
            target,
            value=max(weight, 1.0),
            title=f"Semantic support: {weight:g}",
            color="#A8B3C4",
        )

    net.toggle_physics(physics_enabled)
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -6000,
          "centralGravity": 0.22,
          "springLength": 150,
          "springConstant": 0.035,
          "damping": 0.12
        },
        "minVelocity": 0.75
      },
      "edges": {"smooth": false},
      "nodes": {
        "shape": "dot",
        "font": {"size": 16, "face": "Arial"}
      }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        net.save_graph(temp_file.name)
        html = Path(temp_file.name).read_text(encoding="utf-8")

    components.html(html, height=740, scrolling=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Focus entity", focus_node)
    m2.metric("Entity type", types.get(focus_node, "UNKNOWN"))
    m3.metric("Displayed nodes", display_graph.number_of_nodes())
    m4.metric("Displayed relationships", display_graph.number_of_edges())

    section(
        "Displayed semantic relationships",
        "These are the edges currently shown in the interactive network.",
    )

    display_cols = [
        c for c in [
            source_col,
            source_type_col,
            target_col,
            target_type_col,
            weight_col,
        ]
        if c is not None and c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    json_graph = nx.node_link_data(display_graph)

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.download_button(
            "Download CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{focus_node.replace(' ', '_')}_edges.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        graphml_path = Path(tempfile.gettempdir()) / "stx_lbd_subgraph.graphml"
        nx.write_graphml(display_graph, graphml_path)
        st.download_button(
            "Download GraphML",
            graphml_path.read_bytes(),
            file_name=f"{focus_node.replace(' ', '_')}_subgraph.graphml",
            mime="application/xml",
            use_container_width=True,
        )

    with d3:
        gexf_path = Path(tempfile.gettempdir()) / "stx_lbd_subgraph.gexf"
        nx.write_gexf(display_graph, gexf_path)
        st.download_button(
            "Download GEXF",
            gexf_path.read_bytes(),
            file_name=f"{focus_node.replace(' ', '_')}_subgraph.gexf",
            mime="application/xml",
            use_container_width=True,
        )

    with d4:
        st.download_button(
            "Download JSON",
            json.dumps(json_graph, indent=2).encode("utf-8"),
            file_name=f"{focus_node.replace(' ', '_')}_subgraph.json",
            mime="application/json",
            use_container_width=True,
        )

with tab_node:
    profile_node = st.selectbox(
        "Select an entity for its node profile",
        options=all_nodes,
        key="node_profile_select",
    )

    node_rows = df[
        (df[source_col] == profile_node)
        | (df[target_col] == profile_node)
    ].copy()

    neighbors = set(
        node_rows.loc[node_rows[source_col] == profile_node, target_col]
    ) | set(
        node_rows.loc[node_rows[target_col] == profile_node, source_col]
    )

    degree = len(neighbors)
    weighted_support = float(node_rows[weight_col].sum())

    p1, p2, p3 = st.columns(3)
    p1.metric("Entity", profile_node)
    p2.metric("Entity type", types.get(profile_node, "UNKNOWN"))
    p3.metric("Graph degree", degree)

    st.metric("Total semantic support", f"{weighted_support:g}")

    if node_rows.empty:
        st.info("No semantic relationships were found for this entity.")
    else:
        ranked_neighbors = []
        for neighbor in neighbors:
            pair_rows = node_rows[
                (
                    (node_rows[source_col] == profile_node)
                    & (node_rows[target_col] == neighbor)
                )
                |
                (
                    (node_rows[target_col] == profile_node)
                    & (node_rows[source_col] == neighbor)
                )
            ]
            ranked_neighbors.append(
                {
                    "Connected entity": neighbor,
                    "Entity type": types.get(neighbor, "UNKNOWN"),
                    "Semantic support": float(pair_rows[weight_col].sum()),
                }
            )

        neighbor_df = (
            pd.DataFrame(ranked_neighbors)
            .sort_values("Semantic support", ascending=False)
        )

        section(
            "Top connected entities",
            "Neighbors ranked by cumulative semantic support.",
        )
        st.dataframe(
            neighbor_df,
            use_container_width=True,
            hide_index=True,
        )

with tab_hypotheses:
    hypothesis_node = st.selectbox(
        "Select an entity to retrieve connected hypotheses",
        options=all_nodes,
        key="hypothesis_node_select",
    )

    if engine is None:
        st.warning(
            "The searchable hypothesis database could not be loaded, "
            "so linked hypotheses are unavailable on this page."
        )
    else:
        hypothesis_df = engine.df.copy()
        hypothesis_df.columns = [str(c).strip() for c in hypothesis_df.columns]

        query_norm_col = (
            "Query_Entity_Normalized"
            if "Query_Entity_Normalized" in hypothesis_df.columns
            else "Query_Entity"
        )

        predicted_col = (
            "Predicted_Entity"
            if "Predicted_Entity" in hypothesis_df.columns
            else "Target"
        )

        score_col = (
            "AI_Score"
            if "AI_Score" in hypothesis_df.columns
            else "Score"
        )

        hypothesis_df[score_col] = pd.to_numeric(
            hypothesis_df[score_col],
            errors="coerce",
        )

        normalized_node = hypothesis_node.strip().casefold()

        linked = hypothesis_df[
            hypothesis_df[query_norm_col]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(normalized_node)
        ].copy()

        linked = linked.sort_values(score_col, ascending=False).head(20)

        if linked.empty:
            st.info(
                "No ranked hypotheses were found for this entity "
                "in the searchable hypothesis database."
            )
        else:
            show_cols = [
                c for c in [
                    predicted_col,
                    "Predicted_Entity_Type",
                    "Hypothesis_Class",
                    score_col,
                    "Validation_Status",
                    "Interpretation",
                ]
                if c in linked.columns
            ]

            section(
                f"Top hypotheses for {hypothesis_node}",
                "These predictions come from the same searchable database used by the hypothesis-search page.",
            )

            st.dataframe(
                linked[show_cols],
                use_container_width=True,
                hide_index=True,
            )

            for index, (_, row) in enumerate(linked.iterrows(), start=1):
                predicted = str(row.get(predicted_col, "Unknown"))
                interpretation = str(row.get("Interpretation", "Not available"))
                validation = str(row.get("Validation_Status", "Not assessed"))

                with st.expander(
                    f"{index}. {hypothesis_node} ↔ {predicted}"
                ):
                    e1, e2 = st.columns(2)
                    e1.metric("AI score", f"{float(row.get(score_col, 0)):.3f}")
                    e2.metric("Temporal validation", validation)
                    st.markdown("**Biological interpretation**")
                    st.info(interpretation)
