from pathlib import Path
import sys
import html

import pandas as pd
import streamlit as st


# ============================================================
# PATHS / IMPORT
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CURRENT_DIR)
    )

from query_engine import STXLBD


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="STX-LBD Explorer",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# LIGHT SCIENTIFIC STYLING
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1450px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1 {
    color: #17365d;
    letter-spacing: -0.02em;
}

h2, h3 {
    color: #243b53;
}

.stButton > button {
    border-radius: 5px;
}

.scientific-note {
    color: #64748b;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: -0.4rem;
    margin-bottom: 1.2rem;
}

.method-note {
    background: #f8fafc;
    border-left: 3px solid #466b94;
    padding: 0.65rem 0.85rem;
    margin: 0.5rem 0 1rem 0;
    color: #526579;
    font-size: 0.84rem;
    line-height: 1.55;
}

.bridge-chip {
    display: inline-block;
    padding: 0.18rem 0.48rem;
    margin: 0.12rem 0.08rem 0.12rem 0;
    border-radius: 12px;
    border: 1px solid #dbe3ec;
    background: #f8fafc;
    color: #334155;
    font-size: 0.78rem;
}

.temporal-positive {
    color: #176b4d;
    font-weight: 650;
}

.temporal-open {
    color: #64748b;
    font-weight: 650;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# ENGINE
# ============================================================

@st.cache_resource
def load_engine():
    return STXLBD()


engine = load_engine()


# ============================================================
# HEADER
# ============================================================

st.title("STX-LBD Explorer")

st.markdown(
    """
<div class="scientific-note">
Search graph-ranked candidate relationships involving saxitoxin
biosynthesis genes, taxa, toxins, environmental factors, and
biological processes. Rankings are derived from the historical
pre-2016 dinoflagellate knowledge graph.
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SEARCH
# ============================================================

entity = st.text_input(
    "Search for an entity",
    placeholder=(
        "Examples: sxtA, temperature, "
        "Alexandrium, saxitoxin"
    ),
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns(
    [1, 1.2, 1.4]
)


with col1:

    top_n = st.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )


with col2:

    method = st.selectbox(
        "Graph-ranking method",
        [
            "Adamic–Adar",
            "Jaccard",
            "Node2Vec",
        ],
        index=0,
    )


with col3:

    temporal_filter = st.selectbox(
        "Temporal outcome",
        [
            "All hypotheses",
            "Subsequently represented",
            "Not observed post-2015",
        ],
        index=0,
    )


st.markdown(
    """
<div class="method-note">
<strong>Temporal separation:</strong>
post-2015 literature does not contribute to hypothesis ranking.
It is displayed only as an independent retrospective outcome.
</div>
""",
    unsafe_allow_html=True,
)


search_clicked = st.button(
    "Search",
    type="primary",
)


# ============================================================
# SEARCH RESULTS
# ============================================================

if search_clicked:

    if not entity.strip():

        st.warning(
            "Enter an entity before searching."
        )

        st.stop()


    results = engine.search(
        entity=entity,
        top_n=top_n,
        method=method,
        temporal_filter=temporal_filter,
    )


    if results.empty:

        st.warning(
            f"No hypotheses were found for '{entity}'. "
            "Check the spelling or try another normalized entity."
        )

        st.stop()


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.subheader(
        f"Results for: {entity}"
    )


    display = results[
        [
            "Rank_For_Query",
            "Related_Entity",
            "Related_Entity_Type",
            "Hypothesis_Class",
            "Graph_Score",
            "Temporal_Outcome",
        ]
    ].copy()


    display = display.rename(
        columns={
            "Rank_For_Query": "Rank",
            "Related_Entity": "Related entity",
            "Related_Entity_Type": "Entity type",
            "Hypothesis_Class": "Hypothesis class",
            "Graph_Score": f"{method} score",
            "Temporal_Outcome": "Temporal outcome",
        }
    )


    display[
        f"{method} score"
    ] = (
        pd.to_numeric(
            display[f"{method} score"],
            errors="coerce",
        )
        .round(4)
    )


    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DETAILED HYPOTHESES
    # ========================================================

    st.subheader(
        "Detailed hypotheses"
    )


    for _, row in results.iterrows():

        related_entity = row.get(
            "Related_Entity",
            "Unknown",
        )

        hypothesis_class = (
            str(
                row.get(
                    "Hypothesis_Class",
                    "Unclassified",
                )
            )
            .replace("_", " ")
        )

        graph_score = row.get(
            "Graph_Score",
            float("nan"),
        )

        global_rank = row.get(
            "Global_Graph_Rank",
            float("nan"),
        )

        common_neighbors = row.get(
            "Common_Neighbors",
            float("nan"),
        )

        temporal_outcome = row.get(
            "Temporal_Outcome",
            "Not assessed",
        )


        title = (
            f"{int(row['Rank_For_Query'])}. "
            f"{entity} ↔ {related_entity}"
        )


        with st.expander(title):

            # ------------------------------------------------
            # PRIMARY METRICS
            # ------------------------------------------------

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )


            with metric1:

                if pd.notna(graph_score):

                    st.metric(
                        f"{method} score",
                        f"{float(graph_score):.3f}",
                    )

                else:

                    st.metric(
                        f"{method} score",
                        "NA",
                    )


            with metric2:

                if pd.notna(global_rank):

                    st.metric(
                        f"{method} global rank",
                        f"#{int(global_rank)}",
                    )

                else:

                    st.metric(
                        f"{method} global rank",
                        "NA",
                    )


            with metric3:

                if pd.notna(common_neighbors):

                    st.metric(
                        "Common neighbors",
                        int(common_neighbors),
                    )

                else:

                    st.metric(
                        "Common neighbors",
                        "NA",
                    )


            with metric4:

                st.metric(
                    "Hypothesis class",
                    hypothesis_class,
                )


            # ------------------------------------------------
            # RELATIONSHIP
            # ------------------------------------------------

            st.markdown(
                "**Candidate relationship**"
            )

            st.write(
                f"{entity} ↔ {related_entity}"
            )


            # ------------------------------------------------
            # ENTITY TYPES
            # ------------------------------------------------

            query_type = (
                str(
                    row.get(
                        "Query_Entity_Type",
                        ""
                    )
                )
                .replace("_", " ")
            )

            related_type = (
                str(
                    row.get(
                        "Related_Entity_Type",
                        ""
                    )
                )
                .replace("_", " ")
            )

            if query_type or related_type:

                st.caption(
                    f"{query_type} ↔ {related_type}"
                )


            # ------------------------------------------------
            # BRIDGE CONCEPTS
            # ------------------------------------------------

            bridge_nodes = row.get(
                "Bridge_Nodes",
                "",
            )


            if (
                pd.notna(bridge_nodes)
                and str(bridge_nodes).strip()
            ):

                st.markdown(
                    "**Historical bridge concepts**"
                )

                bridges = [
                    x.strip()
                    for x in str(
                        bridge_nodes
                    ).split(";")
                    if x.strip()
                ]

                bridge_html = ""

                for bridge in bridges[:12]:

                    label = (
                        bridge
                        .replace("_", " ")
                    )

                    bridge_html += (
                        '<span class="bridge-chip">'
                        + html.escape(label)
                        + "</span>"
                    )


                st.markdown(
                    bridge_html,
                    unsafe_allow_html=True,
                )


                if len(bridges) > 12:

                    st.caption(
                        f"+ {len(bridges) - 12} "
                        "additional bridge concepts"
                    )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            interpretation = row.get(
                "Interpretation",
                "",
            )


            if (
                pd.notna(interpretation)
                and str(
                    interpretation
                ).strip()
            ):

                st.markdown(
                    "**Biological interpretation**"
                )

                st.write(
                    interpretation
                )


            # ------------------------------------------------
            # TEMPORAL OUTCOME
            # ------------------------------------------------

            st.markdown(
                "**Temporal outcome**"
            )


            if (
                temporal_outcome
                == "Subsequently represented"
            ):

                st.markdown(
                    """
<div class="temporal-positive">
✓ Subsequently represented in post-2015
dinoflagellate literature
</div>
""",
                    unsafe_allow_html=True,
                )

                st.caption(
                    "This outcome was determined only after "
                    "the historical graph-based ranking and "
                    "did not contribute to the ranking score."
                )


            elif (
                temporal_outcome
                == "Not observed post-2015"
            ):

                st.markdown(
                    """
<div class="temporal-open">
Not observed in the post-2015
dinoflagellate literature used for temporal evaluation
</div>
""",
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Absence from the subsequent literature "
                    "does not constitute biological disproof; "
                    "the relationship remains an open "
                    "literature-derived hypothesis."
                )


            else:

                st.write(
                    temporal_outcome
                )


            # ------------------------------------------------
            # ALTERNATIVE GRAPH RANKINGS
            # ------------------------------------------------

            st.markdown(
                "**Alternative graph rankings**"
            )


            comparison = pd.DataFrame(
                {
                    "Method": [
                        "Adamic–Adar",
                        "Jaccard",
                        "Node2Vec",
                    ],

                    "Score": [
                        row.get(
                            "AdamicAdar_Score"
                        ),
                        row.get(
                            "Jaccard_Score"
                        ),
                        row.get(
                            "Node2Vec_Score"
                        ),
                    ],

                    "Global rank": [
                        row.get(
                            "AdamicAdar_Rank"
                        ),
                        row.get(
                            "Jaccard_Rank"
                        ),
                        row.get(
                            "Node2Vec_Rank"
                        ),
                    ],
                }
            )


            comparison["Score"] = (
                pd.to_numeric(
                    comparison["Score"],
                    errors="coerce",
                )
                .round(4)
            )


            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv_data = (
        results
        .drop(
            columns=[
                "_source_norm",
                "_target_norm",
            ],
            errors="ignore",
        )
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    safe_entity = (
        entity.strip()
        .replace(" ", "_")
        .replace("/", "_")
    )


    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name=(
            f"{safe_entity}_"
            f"stx_lbd_graph_ranked_hypotheses.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "STX-LBD Explorer · Journal proof-of-concept. "
    "Candidate relationships are literature-derived "
    "hypotheses and require independent biological evaluation."
)
