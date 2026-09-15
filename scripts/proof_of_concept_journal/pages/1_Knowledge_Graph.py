from pathlib import Path
import sys
import tempfile

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


KG_FILE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "kg"
    / "dino_pre2016_semantic_edges.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Historical Knowledge Graph | STX-LBD",
    page_icon="🕸️",
    layout="wide",
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Historical Knowledge Graph")

st.write(
    "Explore semantic relationships represented in the "
    "pre-2016 dinoflagellate saxitoxin literature."
)

st.caption(
    "This graph represents the historical knowledge space used "
    "for candidate hypothesis generation and graph-based ranking. "
    "Only information available through 2015 is displayed."
)


# ============================================================
# LOAD HISTORICAL KG
# ============================================================

@st.cache_data
def load_edges():

    if not KG_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {KG_FILE}"
        )

    df = pd.read_csv(KG_FILE)

    source_candidates = [
        "source",
        "Source",
        "source_normalized",
        "Source_Normalized",
    ]

    target_candidates = [
        "target",
        "Target",
        "target_normalized",
        "Target_Normalized",
    ]

    source_type_candidates = [
        "source_type",
        "Source_Type",
    ]

    target_type_candidates = [
        "target_type",
        "Target_Type",
    ]

    weight_candidates = [
        "weight",
        "Weight",
        "edge_weight",
        "Edge_Weight",
        "count",
        "Count",
    ]

    relation_candidates = [
        "relation",
        "Relation",
        "relation_type",
        "Relation_Type",
    ]

    def find_col(candidates):

        for column in candidates:
            if column in df.columns:
                return column

        return None

    source_col = find_col(source_candidates)
    target_col = find_col(target_candidates)

    if source_col is None or target_col is None:
        raise ValueError(
            "Could not identify source/target columns in "
            "dino_pre2016_semantic_edges.csv"
        )

    source_type_col = find_col(
        source_type_candidates
    )

    target_type_col = find_col(
        target_type_candidates
    )

    weight_col = find_col(
        weight_candidates
    )

    relation_col = find_col(
        relation_candidates
    )

    out = pd.DataFrame()

    out["source"] = (
        df[source_col]
        .astype(str)
        .str.strip()
    )

    out["target"] = (
        df[target_col]
        .astype(str)
        .str.strip()
    )

    if source_type_col:
        out["source_type"] = (
            df[source_type_col]
            .fillna("ENTITY")
            .astype(str)
        )
    else:
        out["source_type"] = "ENTITY"

    if target_type_col:
        out["target_type"] = (
            df[target_type_col]
            .fillna("ENTITY")
            .astype(str)
        )
    else:
        out["target_type"] = "ENTITY"

    if weight_col:
        out["weight"] = pd.to_numeric(
            df[weight_col],
            errors="coerce",
        ).fillna(1)
    else:
        out["weight"] = 1

    if relation_col:
        out["relation"] = (
            df[relation_col]
            .fillna("semantic relationship")
            .astype(str)
        )
    else:
        out["relation"] = "semantic relationship"

    out = out[
        (out["source"] != "")
        & (out["target"] != "")
        & (out["source"] != "nan")
        & (out["target"] != "nan")
    ].copy()

    return out


try:
    edges = load_edges()

except Exception as exc:

    st.error(str(exc))
    st.stop()


# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================

G = nx.Graph()

node_types = {}

for _, row in edges.iterrows():

    source = row["source"]
    target = row["target"]

    node_types[source] = str(
        row["source_type"]
    )

    node_types[target] = str(
        row["target_type"]
    )

    weight = float(
        row["weight"]
    )

    relation = str(
        row["relation"]
    )

    if G.has_edge(source, target):

        G[source][target]["weight"] += weight

    else:

        G.add_edge(
            source,
            target,
            weight=weight,
            relation=relation,
        )


# ============================================================
# SUMMARY
# ============================================================

m1, m2, m3 = st.columns(3)

m1.metric(
    "Historical nodes",
    f"{G.number_of_nodes():,}",
)

m2.metric(
    "Historical edges",
    f"{G.number_of_edges():,}",
)

m3.metric(
    "Literature period",
    "≤2015",
)


st.divider()


# ============================================================
# ENTITY SEARCH
# ============================================================

st.subheader("Explore an entity")

all_nodes = sorted(
    list(G.nodes()),
    key=lambda x: str(x).lower(),
)


query = st.text_input(
    "Find an entity",
    placeholder=(
        "Examples: saxitoxin, sxtA, Alexandrium, temperature"
    ),
)


selected = None


if query.strip():

    q = query.strip().lower()

    exact_matches = [
        node
        for node in all_nodes
        if str(node).lower() == q
    ]

    partial_matches = [
        node
        for node in all_nodes
        if q in str(node).lower()
        and node not in exact_matches
    ]

    matches = (
        exact_matches
        + partial_matches
    )

    if matches:

        selected = st.selectbox(
            "Matching entities",
            matches,
            format_func=lambda x: (
                str(x).replace("_", " ")
            ),
        )

    else:

        st.warning(
            f"No historical KG entity matched '{query}'."
        )


# ============================================================
# NETWORK SETTINGS
# ============================================================

c1, c2 = st.columns(2)


with c1:

    neighbor_limit = st.slider(
        "Maximum neighbors",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
        help=(
            "Limits the number of historical connections "
            "displayed around the selected entity."
        ),
    )


with c2:

    min_weight = st.number_input(
        "Minimum edge weight",
        min_value=1.0,
        value=1.0,
        step=1.0,
        help=(
            "Only relationships with at least this historical "
            "edge weight are displayed."
        ),
    )


# ============================================================
# NODE APPEARANCE
# ============================================================

TYPE_STYLE = {

    "TOXIN": {
        "color": "#D95F5F",
        "shape": "dot",
    },

    "SXT_GENE": {
        "color": "#6F63B6",
        "shape": "diamond",
    },

    "ENV_FACTOR": {
        "color": "#4D8F72",
        "shape": "triangle",
    },

    "BIOLOGICAL_PROCESS": {
        "color": "#D9A13B",
        "shape": "dot",
    },

    "DINO_TAXON": {
        "color": "#3D78A8",
        "shape": "hexagon",
    },

    "CYANO_TAXON": {
        "color": "#5B9AA0",
        "shape": "square",
    },

    "DETECTION_METHOD": {
        "color": "#8A7B6A",
        "shape": "dot",
    },

    "ENTITY": {
        "color": "#8B98A7",
        "shape": "dot",
    },
}


def style_for(entity):

    entity_type = node_types.get(
        entity,
        "ENTITY",
    )

    return TYPE_STYLE.get(
        entity_type,
        TYPE_STYLE["ENTITY"],
    )


# ============================================================
# NETWORK RENDERER
# ============================================================

def render_network(
    nodes,
    focus=None,
):

    sub = G.subgraph(
        nodes
    ).copy()

    net = Network(
        height="650px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#243B53",
        directed=False,
    )

    # --------------------------------------------------------
    # ADD NODES
    # --------------------------------------------------------

    for node in sub.nodes():

        style = style_for(
            node
        )

        degree = sub.degree(
            node
        )

        size = min(
            18 + degree * 1.7,
            40,
        )

        border_width = 1

        if node == focus:

            size = max(
                size,
                34,
            )

            border_width = 4

        entity_type = node_types.get(
            node,
            "ENTITY",
        )

        display_name = (
            str(node)
            .replace("_", " ")
        )

        net.add_node(
            node,
            label=display_name,
            title=(
                f"<b>{display_name}</b>"
                f"<br>Type: {entity_type}"
                f"<br>Displayed degree: {degree}"
            ),
            color=style["color"],
            shape=style["shape"],
            size=size,
            borderWidth=border_width,
            font={
                "size": 14,
                "face": "Arial",
                "color": "#243B53",
            },
        )


    # --------------------------------------------------------
    # ADD EDGES
    # --------------------------------------------------------

    for u, v, data in sub.edges(
        data=True
    ):

        weight = float(
            data.get(
                "weight",
                1,
            )
        )

        if weight < min_weight:
            continue

        relation = str(
            data.get(
                "relation",
                "semantic relationship",
            )
        )

        edge_width = min(
            1 + (weight ** 0.5),
            8,
        )

        net.add_edge(
            u,
            v,
            width=edge_width,
            title=(
                f"Relation: "
                f"{relation.replace('_', ' ')}"
                f"<br>Historical weight: {weight:g}"
            ),
            color={
                "color": "#C7D0DA",
                "highlight": "#657786",
                "hover": "#657786",
                "opacity": 0.75,
            },
        )


    # --------------------------------------------------------
    # PHYSICS
    #
    # The graph is allowed to stabilize initially.
    # JavaScript below then disables physics completely.
    # --------------------------------------------------------

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "tooltipDelay": 150
          },

          "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 4
          },

          "edges": {
            "smooth": {
              "enabled": false
            }
          },

          "physics": {
            "enabled": true,
            "solver": "barnesHut",

            "barnesHut": {
              "gravitationalConstant": -12000,
              "centralGravity": 0.15,
              "springLength": 220,
              "springConstant": 0.025,
              "damping": 0.92,
              "avoidOverlap": 1
            },

            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 50,
              "onlyDynamicEdges": false,
              "fit": true
            },

            "minVelocity": 0.75,
            "maxVelocity": 30
          }
        }
        """
    )


    # --------------------------------------------------------
    # GENERATE HTML
    # --------------------------------------------------------

    html = net.generate_html()


    # --------------------------------------------------------
    # FREEZE GRAPH AFTER STABILIZATION
    #
    # This prevents the continuous shaking seen with normal
    # PyVis force-directed networks.
    # --------------------------------------------------------

    freeze_script = """
    <script>

    function freezeSTXNetwork() {

        if (
            typeof network !== "undefined"
            && network !== null
        ) {

            network.once(
                "stabilizationIterationsDone",
                function () {

                    network.setOptions({
                        physics: {
                            enabled: false
                        }
                    });

                    network.fit({
                        animation: {
                            duration: 400,
                            easingFunction: "easeInOutQuad"
                        }
                    });

                }
            );

            /*
             * Safety fallback:
             * Even if the stabilization event is not emitted,
             * stop physics after several seconds.
             */
            setTimeout(
                function () {

                    if (
                        typeof network !== "undefined"
                        && network !== null
                    ) {

                        network.setOptions({
                            physics: {
                                enabled: false
                            }
                        });

                    }

                },
                4500
            );

        }

    }

    freezeSTXNetwork();

    </script>
    """


    html = html.replace(
        "</body>",
        freeze_script
        + "\n</body>",
    )


    # --------------------------------------------------------
    # TEMPORARY HTML FILE
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html",
        mode="w",
        encoding="utf-8",
    ) as tmp:

        tmp.write(
            html
        )

        html_path = (
            tmp.name
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    with open(
        html_path,
        "r",
        encoding="utf-8",
    ) as handle:

        components.html(
            handle.read(),
            height=670,
            scrolling=False,
        )


# ============================================================
# SELECTED ENTITY
# ============================================================

if selected:

    # --------------------------------------------------------
    # FIND HISTORICAL NEIGHBORS
    # --------------------------------------------------------

    neighbors = []

    for neighbor in G.neighbors(
        selected
    ):

        weight = float(
            G[selected][neighbor]
            .get(
                "weight",
                1,
            )
        )

        if weight >= min_weight:

            neighbors.append(
                (
                    neighbor,
                    weight,
                )
            )


    # Strongest historical relationships first.
    neighbors = sorted(
        neighbors,
        key=lambda x: (
            x[1]
        ),
        reverse=True,
    )


    neighbors = neighbors[
        :neighbor_limit
    ]


    neighborhood_nodes = (
        [selected]
        + [
            node
            for node, _
            in neighbors
        ]
    )


    # --------------------------------------------------------
    # ENTITY SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Historical neighborhood"
    )


    e1, e2, e3 = st.columns(
        3
    )


    with e1:

        st.metric(
            "Selected entity",
            str(selected)
            .replace("_", " "),
        )


    with e2:

        st.metric(
            "Entity type",
            node_types.get(
                selected,
                "ENTITY",
            ).replace(
                "_",
                " ",
            ),
        )


    with e3:

        st.metric(
            "Historical degree",
            G.degree(
                selected
            ),
        )


    st.caption(
        "The network below contains the selected entity and "
        "its strongest historical semantic connections."
    )


    # --------------------------------------------------------
    # NETWORK
    # --------------------------------------------------------

    if neighbors:

        render_network(
            neighborhood_nodes,
            focus=selected,
        )

    else:

        st.warning(
            "No historical connections satisfy the selected "
            "minimum edge-weight threshold."
        )


    # --------------------------------------------------------
    # CONNECTION TABLE
    # --------------------------------------------------------

    if neighbors:

        st.markdown(
            "### Historical connections"
        )


        connection_rows = []


        for neighbor, weight in neighbors:

            relation = G[
                selected
            ][neighbor].get(
                "relation",
                "semantic relationship",
            )


            connection_rows.append(
                {
                    "Connected entity":
                        str(neighbor)
                        .replace(
                            "_",
                            " ",
                        ),

                    "Entity type":
                        node_types.get(
                            neighbor,
                            "ENTITY",
                        ).replace(
                            "_",
                            " ",
                        ),

                    "Relationship":
                        str(relation)
                        .replace(
                            "_",
                            " ",
                        ),

                    "Historical weight":
                        weight,
                }
            )


        connection_table = (
            pd.DataFrame(
                connection_rows
            )
        )


        st.dataframe(
            connection_table,
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            "Download displayed connections",
            data=(
                connection_table
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),
            file_name=(
                "STX_LBD_historical_KG_connections.csv"
            ),
            mime="text/csv",
        )


else:

    st.info(
        "Search for an entity above to visualize its "
        "historical semantic neighborhood."
    )


# ============================================================
# LEGEND
# ============================================================

st.divider()

st.markdown(
    "### Semantic entity types"
)


legend_items = [

    (
        "Toxin",
        "#D95F5F",
    ),

    (
        "SXT gene",
        "#6F63B6",
    ),

    (
        "Environmental factor",
        "#4D8F72",
    ),

    (
        "Biological process",
        "#D9A13B",
    ),

    (
        "Dinoflagellate taxon",
        "#3D78A8",
    ),

    (
        "Cyanobacterial taxon",
        "#5B9AA0",
    ),

    (
        "Detection method",
        "#8A7B6A",
    ),
]


legend_cols = st.columns(
    4
)


for i, (
    label,
    color,
) in enumerate(
    legend_items
):

    with legend_cols[
        i % 4
    ]:

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                margin-bottom:8px;
            ">

                <span style="
                    display:inline-block;
                    width:12px;
                    height:12px;
                    border-radius:50%;
                    background:{color};
                    margin-right:7px;
                    border:1px solid #cccccc;
                ">
                </span>

                <span>
                    {label}
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# METHODOLOGICAL NOTE
# ============================================================

st.divider()

st.markdown(
    "### About this graph"
)

st.write(
    "The displayed network is derived exclusively from the "
    "historical dinoflagellate STX semantic knowledge graph "
    "constructed from literature available through 2015. "
    "Post-2015 literature is not used to construct this graph."
)

st.caption(
    "Node size reflects connectivity within the displayed "
    "neighborhood. Edge width reflects historical semantic "
    "edge weight. The interactive layout is stabilized and "
    "then fixed to prevent continuous force-directed movement."
)
