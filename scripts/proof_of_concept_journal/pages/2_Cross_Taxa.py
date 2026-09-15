from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "cross_taxa_temporal"
)

EVALUATION_FILE = (
    DATA_DIR
    / "historical_cyano_transfer_temporal_evaluation.csv"
)

METRICS_FILE = (
    DATA_DIR
    / "cross_taxa_temporal_metrics.csv"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Cross-Taxa Transfer | STX-LBD",
    page_icon="🔄",
    layout="wide",
)

st.title("Cross-Taxa Transfer")

st.write(
    "Explore relationships represented in the historical "
    "cyanobacterial saxitoxin literature by 2015 that were "
    "absent from the dinoflagellate knowledge graph by 2015."
)

st.caption(
    "These historical cyanobacterial relationships were frozen "
    "before evaluation against independent post-2015 "
    "dinoflagellate literature."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Cross-taxa evaluation file not found: "
            f"{EVALUATION_FILE}"
        )

    evaluation = pd.read_csv(
        EVALUATION_FILE
    )

    metrics = None

    if METRICS_FILE.exists():
        metrics = pd.read_csv(
            METRICS_FILE
        )

    return evaluation, metrics


try:
    df, metrics = load_data()

except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# FRIENDLY LABELS
# ============================================================

def pretty(text):

    return (
        str(text)
        .replace("_", " ")
        .strip()
    )


def temporal_label(row):

    if int(row["Temporal_Supported"]) == 1:
        return "Subsequently represented"

    return "Not observed post-2015"


df["Temporal_Outcome"] = df.apply(
    temporal_label,
    axis=1,
)

df["Source_Display"] = (
    df["source"]
    .astype(str)
    .map(pretty)
)

df["Target_Display"] = (
    df["target"]
    .astype(str)
    .map(pretty)
)

df["Relationship"] = (
    df["Source_Display"]
    + " ↔ "
    + df["Target_Display"]
)


# ============================================================
# TEMPORAL DESIGN
# ============================================================

st.markdown("### Temporal design")

c1, arrow1, c2, arrow2, c3 = st.columns(
    [3, 1, 3, 1, 3]
)

with c1:
    st.markdown(
        """
        **Cyanobacteria ≤2015**

        Identify historical STX relationships
        """
    )

with arrow1:
    st.markdown(
        "<div style='text-align:center;"
        "font-size:30px;padding-top:20px;'>→</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        **Absent from Dino ≤2015**

        Freeze cross-taxa candidate relationships
        """
    )

with arrow2:
    st.markdown(
        "<div style='text-align:center;"
        "font-size:30px;padding-top:20px;'>→</div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        **Evaluate in Dino >2015**

        Assess subsequent literature representation
        """
    )


st.divider()


# ============================================================
# SUMMARY
# ============================================================

total = len(df)

represented = int(
    (df["Temporal_Supported"] == 1).sum()
)

not_observed = (
    total - represented
)

representation_rate = (
    represented / total
    if total
    else 0
)


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Historical Cyano-only candidates",
    f"{total:,}",
)

m2.metric(
    "Subsequently represented",
    f"{represented:,}",
)

m3.metric(
    "Not observed post-2015",
    f"{not_observed:,}",
)

m4.metric(
    "Later representation",
    f"{representation_rate:.1%}",
)


st.caption(
    "Subsequent literature representation indicates that the "
    "relationship was later represented in the post-2015 "
    "dinoflagellate literature; it does not constitute "
    "experimental validation."
)


# ============================================================
# EXPLORER
# ============================================================

st.divider()

st.markdown("### Explore cross-taxa relationships")

search = st.text_input(
    "Search entity or relationship",
    placeholder=(
        "Examples: saxitoxin, biosynthesis, "
        "toxin production, nutrient, regulation"
    ),
)


f1, f2 = st.columns(2)

with f1:

    outcome_filter = st.selectbox(
        "Temporal outcome",
        [
            "All",
            "Subsequently represented",
            "Not observed post-2015",
        ],
    )

with f2:

    top_n = st.slider(
        "Number of relationships",
        min_value=10,
        max_value=min(216, len(df)),
        value=min(50, len(df)),
        step=10,
    )


filtered = df.copy()


# Search both endpoints and relation label.
if search.strip():

    q = search.strip().lower()

    mask = (
        filtered["source"]
        .astype(str)
        .str.lower()
        .str.contains(
            q,
            regex=False,
            na=False,
        )
        |
        filtered["target"]
        .astype(str)
        .str.lower()
        .str.contains(
            q,
            regex=False,
            na=False,
        )
        |
        filtered[
            "cyano_historical_relation"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            q,
            regex=False,
            na=False,
        )
    )

    filtered = filtered[mask].copy()


if outcome_filter != "All":

    filtered = filtered[
        filtered["Temporal_Outcome"]
        == outcome_filter
    ].copy()


filtered = (
    filtered
    .sort_values(
        [
            "historical_rank",
            "cyano_pre_weight",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .head(top_n)
)


st.caption(
    f"Showing {len(filtered):,} relationship(s). "
    "Historical ranking is based on pre-2016 "
    "cyanobacterial literature support."
)


# ============================================================
# RESULTS TABLE
# ============================================================

if filtered.empty:

    st.info(
        "No relationships match the selected filters."
    )

else:

    display = filtered[
        [
            "historical_rank",
            "Relationship",
            "source_type",
            "target_type",
            "cyano_pre_weight",
            "Temporal_Outcome",
        ]
    ].copy()

    display.columns = [
        "Historical rank",
        "Candidate relationship",
        "Source type",
        "Target type",
        "Historical Cyano support",
        "Temporal outcome",
    ]

    display["Source type"] = (
        display["Source type"]
        .map(pretty)
    )

    display["Target type"] = (
        display["Target type"]
        .map(pretty)
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RELATIONSHIP DETAILS
# ============================================================

if not filtered.empty:

    st.markdown(
        "### Relationship details"
    )

    relationship_options = (
        filtered["Relationship"]
        .tolist()
    )

    selected_relationship = (
        st.selectbox(
            "Select a relationship",
            relationship_options,
        )
    )

    row = filtered[
        filtered["Relationship"]
        == selected_relationship
    ].iloc[0]


    st.markdown(
        f"#### {row['Relationship']}"
    )


    a, b, c = st.columns(3)

    a.metric(
        "Historical rank",
        f"#{int(row['historical_rank'])}",
    )

    b.metric(
        "Historical Cyano support",
        f"{float(row['cyano_pre_weight']):g}",
    )

    c.metric(
        "Temporal outcome",
        row["Temporal_Outcome"],
    )


    st.markdown(
        "**Historical cyanobacterial evidence**"
    )

    st.write(
        f"Relation type: "
        f"`{pretty(row['cyano_historical_relation'])}`"
    )

    st.write(
        f"Entity types: "
        f"`{pretty(row['source_type'])}` → "
        f"`{pretty(row['target_type'])}`"
    )


    if int(row["Temporal_Supported"]) == 1:

        st.success(
            "This relationship was subsequently represented "
            "in the post-2015 dinoflagellate literature."
        )

        if pd.notna(
            row["dino_post_relation"]
        ):

            st.write(
                "**Post-2015 dinoflagellate relation:** "
                f"`{pretty(row['dino_post_relation'])}`"
            )

        if pd.notna(
            row["dino_post_weight"]
        ):

            st.write(
                "**Post-2015 dinoflagellate support:** "
                f"{float(row['dino_post_weight']):g}"
            )

    else:

        st.info(
            "This relationship was not observed in the "
            "post-2015 dinoflagellate literature used for "
            "temporal evaluation. This absence should not be "
            "interpreted as biological rejection of the "
            "candidate relationship."
        )


# ============================================================
# RANKING PERFORMANCE
# ============================================================

if metrics is not None:

    st.divider()

    st.markdown(
        "### Historical cross-taxa ranking performance"
    )

    st.caption(
        "Candidates are ranked using only historical "
        "cyanobacterial literature support. Random expectation "
        "was estimated by permutation."
    )


    metric_display = (
        metrics.copy()
    )

    metric_display["Precision"] = (
        metric_display["Precision"]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    metric_display[
        "Random_Mean_Precision"
    ] = (
        metric_display[
            "Random_Mean_Precision"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    metric_display[
        "Enrichment_vs_Random"
    ] = (
        metric_display[
            "Enrichment_vs_Random"
        ]
        .map(
            lambda x: f"{x:.2f}×"
        )
    )

    metric_display[
        "Empirical_P"
    ] = (
        metric_display[
            "Empirical_P"
        ]
        .map(
            lambda x: (
                "<0.0001"
                if x < 0.0001
                else f"{x:.4f}"
            )
        )
    )


    metric_display = metric_display[
        [
            "K",
            "Hits",
            "Precision",
            "Random_Mean_Precision",
            "Enrichment_vs_Random",
            "Empirical_P",
        ]
    ]

    metric_display.columns = [
        "K",
        "Hits",
        "Precision@K",
        "Random mean",
        "Enrichment",
        "Empirical P",
    ]

    st.dataframe(
        metric_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.download_button(
    "Download displayed relationships",
    data=filtered.to_csv(
        index=False
    ).encode("utf-8"),
    file_name=(
        "STX_LBD_historical_cross_taxa_relationships.csv"
    ),
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Cross-taxa analysis uses cyanobacterial literature "
    "through 2015 as the historical knowledge source. "
    "Post-2015 dinoflagellate literature is used only for "
    "subsequent temporal evaluation."
)
