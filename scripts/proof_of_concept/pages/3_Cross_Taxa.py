
from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import tempfile

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


# ============================================================
# PATHS / IMPORTS
# ============================================================

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import apply_global_style, hero, render_sidebar, section


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="STX-LBD Cross-Taxa Explorer",
    page_icon="🧬",
    layout="wide",
)

apply_global_style()
render_sidebar()


# ============================================================
# FILE LOCATORS
# ============================================================

KG_DIR = PROJECT_ROOT / "FINAL_WORKSPACE" / "kg"


def locate_file(exact_name: str, fallback_patterns: list[str]) -> Path | None:
    exact = KG_DIR / exact_name
    if exact.exists():
        return exact

    if KG_DIR.exists():
        for pattern in fallback_patterns:
            matches = sorted(KG_DIR.glob(pattern))
            if matches:
                return matches[0]

    workspace = PROJECT_ROOT / "FINAL_WORKSPACE"

    if workspace.exists():
        exact_matches = sorted(workspace.rglob(exact_name))
        if exact_matches:
            return exact_matches[0]

        for pattern in fallback_patterns:
            matches = sorted(workspace.rglob(pattern))
            if matches:
                return matches[0]

    return None


FILES = {
    "dino_all": locate_file(
        "dino_all_semantic_edges.csv",
        ["*dino*all*semantic*edge*.csv"],
    ),
    "cyano_all": locate_file(
        "cyano_all_semantic_edges.csv",
        ["*cyano*all*semantic*edge*.csv"],
    ),
    "dino_pre2016": locate_file(
        "dino_pre2016_semantic_edges.csv",
        ["*dino*pre*2016*semantic*edge*.csv"],
    ),
    "dino_post2015": locate_file(
        "dino_post2015_semantic_edges.csv",
        ["*dino*post*2015*semantic*edge*.csv"],
    ),
}

missing_required = [
    name
    for name in ["dino_all", "cyano_all"]
    if FILES[name] is None
]

if missing_required:
    st.error(
        "The Cross-Taxa Explorer requires the final dinoflagellate and "
        "cyanobacterial knowledge graphs. Missing: "
        + ", ".join(missing_required)
    )
    st.stop()


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def find_col(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {
        str(column).strip().casefold(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.casefold()

        if key in lookup:
            return lookup[key]

    return None


def canonical(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().casefold().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def readable_type(value: object) -> str:
    text = str(value).strip()

    labels = {
        "SXT_GENE": "STX gene",
        "TOXIN": "Toxin",
        "DINO_TAXON": "Dinoflagellate taxon",
        "CYANO_TAXON": "Cyanobacterial taxon",
        "ENV_FACTOR": "Environmental factor",
        "BIOLOGICAL_PROCESS": "Biological process",
        "DETECTION_METHOD": "Detection method",
    }

    return labels.get(text, text.replace("_", " ").title())


def split_support_documents(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    docs = []
    seen = set()

    for item in re.split(r"[;,|]", str(value)):
        doc_id = item.strip()

        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)

    return docs


def prepare_graph(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    source_col = find_col(frame, "source", "Source")
    target_col = find_col(frame, "target", "Target")
    source_type_col = find_col(frame, "source_type", "Source_Type")
    target_type_col = find_col(frame, "target_type", "Target_Type")
    relation_col = find_col(frame, "relation", "Relation")
    weight_col = find_col(frame, "weight", "Weight", "count", "Count")
    support_col = find_col(frame, "support_documents", "Support_Documents")
    first_year_col = find_col(frame, "first_year", "First_Year")
    last_year_col = find_col(frame, "last_year", "Last_Year")

    if source_col is None or target_col is None:
        raise ValueError(
            f"{label} graph must contain source and target columns."
        )

    out = frame.dropna(
        subset=[source_col, target_col]
    ).copy()

    out["source"] = (
        out[source_col]
        .astype(str)
        .str.strip()
    )

    out["target"] = (
        out[target_col]
        .astype(str)
        .str.strip()
    )

    out["source_type"] = (
        out[source_type_col]
        .astype(str)
        .str.strip()
        if source_type_col
        else "UNKNOWN"
    )

    out["target_type"] = (
        out[target_type_col]
        .astype(str)
        .str.strip()
        if target_type_col
        else "UNKNOWN"
    )

    out["relation"] = (
        out[relation_col]
        .astype(str)
        .str.strip()
        if relation_col
        else ""
    )

    out["weight"] = (
        pd.to_numeric(
            out[weight_col],
            errors="coerce",
        ).fillna(1.0)
        if weight_col
        else 1.0
    )

    out["support_documents"] = (
        out[support_col]
        if support_col
        else ""
    )

    out["support_count"] = (
        out["support_documents"]
        .map(
            lambda value: len(
                split_support_documents(value)
            )
        )
    )

    out["first_year"] = (
        pd.to_numeric(
            out[first_year_col],
            errors="coerce",
        )
        if first_year_col
        else pd.NA
    )

    out["last_year"] = (
        pd.to_numeric(
            out[last_year_col],
            errors="coerce",
        )
        if last_year_col
        else pd.NA
    )

    out["_source_norm"] = (
        out["source"].map(canonical)
    )

    out["_target_norm"] = (
        out["target"].map(canonical)
    )

    out["_pair"] = out.apply(
        lambda row: "||".join(
            sorted(
                [
                    row["_source_norm"],
                    row["_target_norm"],
                ]
            )
        ),
        axis=1,
    )

    # Collapse repeated semantic rows while preserving normalized columns.
    collapsed = (
        out.groupby(
            "_pair",
            as_index=False,
        )
        .agg(
            source=("source", "first"),
            target=("target", "first"),
            _source_norm=("_source_norm", "first"),
            _target_norm=("_target_norm", "first"),
            source_type=("source_type", "first"),
            target_type=("target_type", "first"),
            weight=("weight", "sum"),
            support_count=("support_count", "sum"),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            relation=(
                "relation",
                lambda s: "; ".join(
                    sorted(
                        set(
                            x
                            for x in s.astype(str)
                            if x
                            and x != "nan"
                        )
                    )
                ),
            ),
        )
    )

    collapsed["dataset"] = label

    return collapsed


# ============================================================
# LOAD ALL GRAPHS
# ============================================================

graphs: dict[str, pd.DataFrame] = {}

for name, path in FILES.items():
    if path is not None:
        graphs[name] = prepare_graph(
            load_csv(str(path)),
            name,
        )

dino_all = graphs["dino_all"]
cyano_all = graphs["cyano_all"]
dino_pre = graphs.get("dino_pre2016")
dino_post = graphs.get("dino_post2015")


# ============================================================
# CORE CROSS-TAXA SETS
# ============================================================

dino_pairs = set(dino_all["_pair"])
cyano_pairs = set(cyano_all["_pair"])

shared_pairs = dino_pairs & cyano_pairs
dino_only_pairs = dino_pairs - cyano_pairs
cyano_only_pairs = cyano_pairs - dino_pairs

pre_pairs = (
    set(dino_pre["_pair"])
    if dino_pre is not None
    else set()
)

post_pairs = (
    set(dino_post["_pair"])
    if dino_post is not None
    else set()
)

cyano_absent_pre = (
    cyano_pairs - pre_pairs
    if dino_pre is not None
    else set()
)

later_supported = (
    cyano_absent_pre & post_pairs
    if dino_post is not None
    else set()
)


# ============================================================
# PAGE HEADER
# ============================================================

hero(
    "🧬 Cross-Taxa Comparative Biology Explorer",
    "Compare dinoflagellate and cyanobacterial saxitoxin biology, identify "
    "shared and lineage-specific semantic relationships, and examine "
    "cyanobacterial knowledge signals that later appear in dinoflagellate literature.",
)

st.info(
    "This module compares literature-derived semantic relationships. "
    "Shared relationships indicate cross-taxa convergence or conservation at the "
    "knowledge level; they do not by themselves demonstrate common ancestry, "
    "horizontal gene transfer, or identical regulatory mechanisms."
)


# ============================================================
# SUMMARY
# ============================================================

section(
    "Comparative STX knowledge landscape",
    "High-level comparison of the final dinoflagellate and cyanobacterial semantic graphs.",
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Dinoflagellate relationships",
    f"{len(dino_pairs):,}",
)

m2.metric(
    "Cyanobacterial relationships",
    f"{len(cyano_pairs):,}",
)

m3.metric(
    "Shared relationships",
    f"{len(shared_pairs):,}",
)

m4.metric(
    "Lineage-specific relationships",
    f"{len(dino_only_pairs) + len(cyano_only_pairs):,}",
)

if dino_pre is not None and dino_post is not None:
    p1, p2 = st.columns(2)

    p1.metric(
        "Cyano signals absent from pre-2016 dino KG",
        f"{len(cyano_absent_pre):,}",
    )

    p2.metric(
        "Later represented in post-2015 dino KG",
        f"{len(later_supported):,}",
    )


# ============================================================
# MAIN TABS
# ============================================================

tab_entity, tab_transfer, tab_shared, tab_overview = st.tabs(
    [
        "Compare an entity",
        "Cross-taxa transfer signals",
        "Shared biology network",
        "Global overview",
    ]
)


# ============================================================
# TAB 1 — ENTITY COMPARISON
# ============================================================

with tab_entity:

    all_entities = sorted(
        set(dino_all["source"])
        | set(dino_all["target"])
        | set(cyano_all["source"])
        | set(cyano_all["target"])
    )

    selected_entity = st.selectbox(
        "Search a gene, toxin, environmental factor, process, or taxon",
        options=all_entities,
        placeholder=(
            "Examples: sxtA, sxtG, saxitoxin, temperature, nitrogen..."
        ),
    )

    entity_norm = canonical(
        selected_entity
    )

    def connected_edges(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        return frame[
            frame["_source_norm"].eq(
                entity_norm
            )
            |
            frame["_target_norm"].eq(
                entity_norm
            )
        ].copy()

    dino_entity = connected_edges(
        dino_all
    )

    cyano_entity = connected_edges(
        cyano_all
    )

    dino_entity_pairs = set(
        dino_entity["_pair"]
    )

    cyano_entity_pairs = set(
        cyano_entity["_pair"]
    )

    shared_for_entity = (
        dino_entity_pairs
        & cyano_entity_pairs
    )

    dino_only_for_entity = (
        dino_entity_pairs
        - cyano_entity_pairs
    )

    cyano_only_for_entity = (
        cyano_entity_pairs
        - dino_entity_pairs
    )

    st.markdown(
        f"### Comparative profile: {selected_entity}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Shared associations",
        len(shared_for_entity),
    )

    c2.metric(
        "Dinoflagellate-specific",
        len(dino_only_for_entity),
    )

    c3.metric(
        "Cyanobacteria-specific",
        len(cyano_only_for_entity),
    )

    def partner_from_row(
        row: pd.Series,
    ) -> tuple[str, str]:
        if (
            row["_source_norm"]
            == entity_norm
        ):
            return (
                row["target"],
                row["target_type"],
            )

        return (
            row["source"],
            row["source_type"],
        )

    rows = []

    for frame, lineage in [
        (
            dino_entity,
            "Dinoflagellate",
        ),
        (
            cyano_entity,
            "Cyanobacteria",
        ),
    ]:
        for _, row in frame.iterrows():
            partner, partner_type = (
                partner_from_row(row)
            )

            if row["_pair"] in shared_for_entity:
                status = "Shared across taxa"

            elif lineage == "Dinoflagellate":
                status = "Dinoflagellate-specific"

            else:
                status = "Cyanobacteria-specific"

            rows.append(
                {
                    "Connected entity": partner,
                    "Entity type": readable_type(
                        partner_type
                    ),
                    "Lineage": lineage,
                    "Cross-taxa status": status,
                    "Semantic support": row[
                        "weight"
                    ],
                    "Supporting documents": row[
                        "support_count"
                    ],
                    "Relation": row[
                        "relation"
                    ],
                }
            )

    comparison_df = pd.DataFrame(
        rows
    )

    if comparison_df.empty:
        st.info(
            "No semantic relationships were found for this entity."
        )

    else:
        shared_df = comparison_df[
            comparison_df[
                "Cross-taxa status"
            ].eq(
                "Shared across taxa"
            )
        ].copy()

        dino_specific_df = comparison_df[
            comparison_df[
                "Cross-taxa status"
            ].eq(
                "Dinoflagellate-specific"
            )
        ].copy()

        cyano_specific_df = comparison_df[
            comparison_df[
                "Cross-taxa status"
            ].eq(
                "Cyanobacteria-specific"
            )
        ].copy()

        st.markdown(
            "#### Shared biological associations"
        )

        if shared_df.empty:
            st.caption(
                "No shared associations were identified."
            )

        else:
            st.dataframe(
                shared_df.sort_values(
                    "Semantic support",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

        left, right = st.columns(2)

        with left:
            st.markdown(
                "#### Dinoflagellate-specific associations"
            )

            if dino_specific_df.empty:
                st.caption(
                    "No lineage-specific associations."
                )

            else:
                st.dataframe(
                    dino_specific_df.sort_values(
                        "Semantic support",
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with right:
            st.markdown(
                "#### Cyanobacteria-specific associations"
            )

            if cyano_specific_df.empty:
                st.caption(
                    "No lineage-specific associations."
                )

            else:
                st.dataframe(
                    cyano_specific_df.sort_values(
                        "Semantic support",
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # Conservative automated interpretation.
        if len(shared_for_entity) > 0:
            st.success(
                f"{selected_entity} participates in {len(shared_for_entity)} "
                "semantic relationship(s) represented in both lineages. "
                "This indicates cross-taxa convergence or conservation in the "
                "published STX knowledge base and may highlight biologically "
                "important shared mechanisms for further investigation."
            )

        elif (
            len(dino_only_for_entity)
            > 0
            and len(cyano_only_for_entity)
            > 0
        ):
            st.info(
                f"{selected_entity} is represented in both lineages, but the "
                "specific semantic neighborhoods differ. This pattern may reflect "
                "lineage-specific biology, different research emphasis, or "
                "divergent ecological and molecular mechanisms."
            )

        st.download_button(
            "Download comparative profile",
            comparison_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                f"{selected_entity.replace(' ', '_')}"
                "_cross_taxa_profile.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# TAB 2 — TRANSFER SIGNALS
# ============================================================

with tab_transfer:

    if (
        dino_pre is None
        or dino_post is None
    ):
        st.warning(
            "This view requires both "
            "`dino_pre2016_semantic_edges.csv` and "
            "`dino_post2015_semantic_edges.csv`."
        )

    else:
        st.markdown(
            """
            ### Cyanobacteria-to-dinoflagellate knowledge signals

            These relationships are represented in the cyanobacterial STX
            knowledge graph but were absent from the pre-2016 dinoflagellate
            graph. Relationships that later appear in the post-2015
            dinoflagellate graph are highlighted as **later-supported
            cross-taxa signals**.

            Because the cyanobacterial graph represents the full literature
            corpus, this is a comparative transfer/convergence analysis rather
            than a strictly prospective prediction.
            """
        )

        candidate_df = cyano_all[
            cyano_all["_pair"].isin(
                cyano_absent_pre
            )
        ].copy()

        candidate_df[
            "Later_Dinoflagellate_Support"
        ] = candidate_df[
            "_pair"
        ].isin(
            later_supported
        )

        post_lookup = (
            dino_post[
                [
                    "_pair",
                    "weight",
                    "support_count",
                    "first_year",
                    "last_year",
                ]
            ]
            .rename(
                columns={
                    "weight": "Post2015_Dino_Weight",
                    "support_count": "Post2015_Dino_Documents",
                    "first_year": "Post2015_First_Year",
                    "last_year": "Post2015_Last_Year",
                }
            )
        )

        candidate_df = (
            candidate_df.merge(
                post_lookup,
                on="_pair",
                how="left",
            )
        )

        f1, f2 = st.columns(2)

        with f1:
            status_filter = st.selectbox(
                "Signal status",
                [
                    "All candidates",
                    "Later supported in dinoflagellates",
                    "Still cyanobacteria-only",
                ],
            )

        with f2:
            entity_filter = st.text_input(
                "Filter by entity",
                placeholder=(
                    "Examples: sxtA, light, temperature, nitrogen"
                ),
            )

        if (
            status_filter
            == "Later supported in dinoflagellates"
        ):
            candidate_df = candidate_df[
                candidate_df[
                    "Later_Dinoflagellate_Support"
                ]
            ]

        elif (
            status_filter
            == "Still cyanobacteria-only"
        ):
            candidate_df = candidate_df[
                ~candidate_df[
                    "Later_Dinoflagellate_Support"
                ]
            ]

        if entity_filter.strip():
            term = canonical(
                entity_filter
            )

            candidate_df = candidate_df[
                candidate_df[
                    "_source_norm"
                ].str.contains(
                    re.escape(term),
                    na=False,
                )
                |
                candidate_df[
                    "_target_norm"
                ].str.contains(
                    re.escape(term),
                    na=False,
                )
            ]

        candidate_df = (
            candidate_df.sort_values(
                [
                    "Later_Dinoflagellate_Support",
                    "weight",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        t1, t2, t3 = st.columns(3)

        t1.metric(
            "Displayed candidates",
            len(candidate_df),
        )

        t2.metric(
            "Later-supported",
            int(
                candidate_df[
                    "Later_Dinoflagellate_Support"
                ].sum()
            )
            if not candidate_df.empty
            else 0,
        )

        t3.metric(
            "Still cyano-only",
            int(
                (
                    ~candidate_df[
                        "Later_Dinoflagellate_Support"
                    ]
                ).sum()
            )
            if not candidate_df.empty
            else 0,
        )

        display_cols = [
            "source",
            "target",
            "source_type",
            "target_type",
            "weight",
            "support_count",
            "Later_Dinoflagellate_Support",
            "Post2015_Dino_Weight",
            "Post2015_Dino_Documents",
            "Post2015_First_Year",
            "Post2015_Last_Year",
        ]

        st.dataframe(
            candidate_df[
                [
                    c
                    for c in display_cols
                    if c
                    in candidate_df.columns
                ]
            ].head(
                300
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download cross-taxa candidate signals",
            candidate_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                "STX_LBD_cross_taxa_transfer_signals.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# TAB 3 — SHARED BIOLOGY NETWORK
# ============================================================

with tab_shared:

    shared_dino = (
        dino_all[
            dino_all[
                "_pair"
            ].isin(
                shared_pairs
            )
        ]
        .copy()
    )

    shared_cyano_support = (
        cyano_all[
            [
                "_pair",
                "weight",
                "support_count",
            ]
        ]
        .rename(
            columns={
                "weight": "Cyano_Weight",
                "support_count": "Cyano_Documents",
            }
        )
    )

    shared_df = (
        shared_dino.merge(
            shared_cyano_support,
            on="_pair",
            how="left",
        )
        .rename(
            columns={
                "weight": "Dino_Weight",
                "support_count": "Dino_Documents",
            }
        )
    )

    shared_df[
        "Combined_Support"
    ] = (
        shared_df[
            "Dino_Weight"
        ].fillna(
            0
        )
        +
        shared_df[
            "Cyano_Weight"
        ].fillna(
            0
        )
    )

    shared_entities = sorted(
        set(
            shared_df[
                "source"
            ]
        )
        |
        set(
            shared_df[
                "target"
            ]
        )
    )

    n1, n2 = st.columns(
        [2, 1]
    )

    with n1:
        network_focus = st.selectbox(
            "Focus entity",
            options=[
                "All shared relationships"
            ]
            + shared_entities,
        )

    with n2:
        max_edges = st.slider(
            "Maximum edges",
            min_value=25,
            max_value=400,
            value=100,
            step=25,
        )

    network_df = (
        shared_df.copy()
    )

    if (
        network_focus
        != "All shared relationships"
    ):
        focus_norm = canonical(
            network_focus
        )

        network_df = network_df[
            network_df[
                "_source_norm"
            ].eq(
                focus_norm
            )
            |
            network_df[
                "_target_norm"
            ].eq(
                focus_norm
            )
        ]

    network_df = (
        network_df.sort_values(
            "Combined_Support",
            ascending=False,
        )
        .head(
            max_edges
        )
    )

    if network_df.empty:
        st.info(
            "No shared network relationships were found."
        )

    else:
        graph = nx.Graph()
        node_types = {}

        for _, row in network_df.iterrows():
            graph.add_edge(
                row["source"],
                row["target"],
                dino_weight=float(
                    row["Dino_Weight"]
                ),
                cyano_weight=float(
                    row["Cyano_Weight"]
                ),
                combined_support=float(
                    row["Combined_Support"]
                ),
            )

            node_types[
                row["source"]
            ] = row[
                "source_type"
            ]

            node_types[
                row["target"]
            ] = row[
                "target_type"
            ]

        type_colors = {
            "SXT_GENE": "#287DB2",
            "TOXIN": "#E76F00",
            "DINO_TAXON": "#159A73",
            "CYANO_TAXON": "#58AFCC",
            "ENV_FACTOR": "#F2A900",
            "BIOLOGICAL_PROCESS": "#CF6EA3",
            "DETECTION_METHOD": "#8C8C8C",
        }

        net = Network(
            height="720px",
            width="100%",
            bgcolor="#FFFFFF",
            font_color="#1F2937",
            notebook=False,
            cdn_resources="in_line",
        )

        for node in graph.nodes():
            node_type = node_types.get(
                node,
                "UNKNOWN",
            )

            net.add_node(
                node,
                label=node,
                title=(
                    f"{node}"
                    f"<br>Type: {readable_type(node_type)}"
                ),
                color=type_colors.get(
                    node_type,
                    "#6B7280",
                ),
                size=(
                    30
                    if node
                    == network_focus
                    else 15
                    + min(
                        graph.degree(
                            node
                        ),
                        10,
                    )
                ),
                borderWidth=(
                    4
                    if node
                    == network_focus
                    else 1
                ),
            )

        for (
            source,
            target,
            data,
        ) in graph.edges(
            data=True
        ):
            net.add_edge(
                source,
                target,
                value=max(
                    data[
                        "combined_support"
                    ],
                    1,
                ),
                title=(
                    "Dinoflagellate support: "
                    f"{data['dino_weight']:g}"
                    "<br>Cyanobacterial support: "
                    f"{data['cyano_weight']:g}"
                ),
                color="#9AA9BD",
            )

        net.set_options(
            """
            {
              "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -6000,
                  "centralGravity": 0.22,
                  "springLength": 155,
                  "springConstant": 0.035,
                  "damping": 0.12
                },
                "minVelocity": 0.75
              },
              "edges": {
                "smooth": false
              },
              "nodes": {
                "shape": "dot",
                "font": {
                  "size": 16,
                  "face": "Arial"
                }
              }
            }
            """
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".html",
        ) as temp_file:
            net.save_graph(
                temp_file.name
            )

            html = Path(
                temp_file.name
            ).read_text(
                encoding="utf-8"
            )

        components.html(
            html,
            height=740,
            scrolling=False,
        )

        s1, s2 = st.columns(2)

        s1.metric(
            "Displayed shared nodes",
            graph.number_of_nodes(),
        )

        s2.metric(
            "Displayed shared relationships",
            graph.number_of_edges(),
        )

        json_graph = nx.node_link_data(
            graph
        )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Download shared relationships CSV",
                network_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                ),
                file_name=(
                    "STX_LBD_shared_cross_taxa_relationships.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download network JSON",
                json.dumps(
                    json_graph,
                    indent=2,
                ).encode(
                    "utf-8"
                ),
                file_name=(
                    "STX_LBD_shared_cross_taxa_network.json"
                ),
                mime="application/json",
                use_container_width=True,
            )


# ============================================================
# TAB 4 — GLOBAL OVERVIEW
# ============================================================

with tab_overview:

    overview_df = pd.DataFrame(
        {
            "Category": [
                "Shared across taxa",
                "Dinoflagellate-specific",
                "Cyanobacteria-specific",
            ],
            "Relationships": [
                len(shared_pairs),
                len(dino_only_pairs),
                len(cyano_only_pairs),
            ],
        }
    )

    st.bar_chart(
        overview_df.set_index(
            "Category"
        ),
        horizontal=True,
    )

    top_shared = (
        dino_all[
            dino_all[
                "_pair"
            ].isin(
                shared_pairs
            )
        ]
        .merge(
            cyano_all[
                [
                    "_pair",
                    "weight",
                    "support_count",
                ]
            ].rename(
                columns={
                    "weight": "Cyano_Weight",
                    "support_count": "Cyano_Documents",
                }
            ),
            on="_pair",
            how="left",
        )
        .rename(
            columns={
                "weight": "Dino_Weight",
                "support_count": "Dino_Documents",
            }
        )
    )

    top_shared[
        "Combined_Support"
    ] = (
        top_shared[
            "Dino_Weight"
        ].fillna(
            0
        )
        +
        top_shared[
            "Cyano_Weight"
        ].fillna(
            0
        )
    )

    section(
        "Strongest shared semantic relationships",
        "Relationships represented in both lineages, ranked by combined semantic support.",
    )

    st.dataframe(
        top_shared.sort_values(
            "Combined_Support",
            ascending=False,
        )[
            [
                "source",
                "target",
                "source_type",
                "target_type",
                "Dino_Weight",
                "Cyano_Weight",
                "Dino_Documents",
                "Cyano_Documents",
                "Combined_Support",
            ]
        ].head(
            100
        ),
        use_container_width=True,
        hide_index=True,
    )
