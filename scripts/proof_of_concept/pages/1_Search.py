
from pathlib import Path
import sys
import urllib.parse

import pandas as pd
import streamlit as st


# ============================================================
# PATHS / IMPORTS
# ============================================================

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from query_engine import STXLBD
from biological_interpretation import generate_biological_interpretation
from ui import apply_global_style, hero, render_sidebar, section


# ============================================================
# PAGE CONFIG
# ============================================================


apply_global_style()
render_sidebar()


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_resource
def load_engine():
    return STXLBD()


@st.cache_data
def load_validation_evidence():
    evidence_path = (
        PROJECT_ROOT
        / "FINAL_WORKSPACE"
        / "proof_of_concept"
        / "hypothesis_validation_evidence.csv"
    )

    if not evidence_path.exists():
        return None, evidence_path

    evidence = pd.read_csv(evidence_path)
    evidence.columns = [str(c).strip() for c in evidence.columns]

    if "Year" in evidence.columns:
        evidence["Year"] = pd.to_numeric(
            evidence["Year"],
            errors="coerce",
        )

    return evidence, evidence_path


try:
    engine = load_engine()
except FileNotFoundError:
    st.error(
        "The searchable STX-LBD database was not found at "
        "`FINAL_WORKSPACE/proof_of_concept/searchable_hypotheses.csv`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the STX-LBD query engine: {exc}")
    st.stop()

evidence_df, evidence_path = load_validation_evidence()

df = engine.df.copy()
df.columns = [str(c).strip() for c in df.columns]


# ============================================================
# HELPERS
# ============================================================

def first_column(frame, *names):
    return next((name for name in names if name in frame.columns), None)


def text_value(value, fallback="Not available"):
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()

    if not text or text in {"-", "nan", "None"}:
        return fallback

    return text


def score_value(value):
    try:
        if value is None or pd.isna(value):
            return "Not available"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return text_value(value)


def canonical_entity(value):
    if value is None or pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .casefold()
        .replace("_", " ")
    )


def normalized_pair(entity_a, entity_b):
    return "||".join(
        sorted(
            [
                canonical_entity(entity_a),
                canonical_entity(entity_b),
            ]
        )
    )


def doi_url(doi):
    doi = text_value(doi, "")
    if not doi:
        return None

    doi = doi.replace("https://doi.org/", "").strip()
    return f"https://doi.org/{urllib.parse.quote(doi, safe='/:().-_')}"


def pmid_url(pmid):
    pmid = text_value(pmid, "")
    if not pmid:
        return None

    # Clean values such as 38705616.0 if pandas interpreted as numeric.
    if pmid.endswith(".0"):
        pmid = pmid[:-2]

    return f"https://pubmed.ncbi.nlm.nih.gov/{urllib.parse.quote(pmid)}/"


def evidence_strength(publication_count):
    if publication_count >= 5:
        return "Strong literature support"
    if publication_count >= 2:
        return "Moderate literature support"
    if publication_count == 1:
        return "Emerging literature support"
    return "No traceable literature support"


def get_hypothesis_evidence(query_entity, predicted_entity):
    if evidence_df is None or evidence_df.empty:
        return pd.DataFrame()

    if (
        "Query_Entity" not in evidence_df.columns
        or "Predicted_Entity" not in evidence_df.columns
    ):
        return pd.DataFrame()

    wanted_pair = normalized_pair(
        query_entity,
        predicted_entity,
    )

    pair_series = evidence_df.apply(
        lambda row: normalized_pair(
            row.get("Query_Entity", ""),
            row.get("Predicted_Entity", ""),
        ),
        axis=1,
    )

    evidence = evidence_df[
        pair_series.eq(wanted_pair)
    ].copy()

    if "Evidence_Match" in evidence.columns:
        evidence = evidence[
            evidence["Evidence_Match"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("matched")
        ].copy()

    if "Document_ID" in evidence.columns:
        evidence = evidence.drop_duplicates(
            subset=["Document_ID"],
            keep="first",
        )

    if "Year" in evidence.columns:
        evidence = evidence.sort_values(
            "Year",
            ascending=False,
            na_position="last",
        )

    return evidence


def render_publication(row, number):
    title = text_value(
        row.get("Title"),
        "Untitled supporting publication",
    )
    journal = text_value(row.get("Journal"), "")
    year = row.get("Year")

    if pd.notna(year):
        try:
            year_display = str(int(float(year)))
        except Exception:
            year_display = text_value(year, "")
    else:
        year_display = ""

    metadata_bits = [
        bit
        for bit in [journal, year_display]
        if bit
    ]

    st.markdown(f"**{number}. {title}**")

    if metadata_bits:
        st.caption(" · ".join(metadata_bits))

    link_parts = []

    doi = text_value(row.get("DOI"), "")
    pmid = text_value(row.get("PMID"), "")
    url = text_value(row.get("URL"), "")

    if doi:
        link_parts.append(
            f"[DOI]({doi_url(doi)})"
        )

    if pmid:
        link_parts.append(
            f"[PubMed]({pmid_url(pmid)})"
        )

    if url and url.startswith(("http://", "https://")):
        link_parts.append(
            f"[Source]({url})"
        )

    if link_parts:
        st.markdown(" · ".join(link_parts))


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

for numeric_col in [
    "AI_Score",
    "Node2Vec_Integrated_Score",
    "Embedding_Integrated_Score",
    "ML_Probability",
]:
    if numeric_col in df.columns:
        df[numeric_col] = pd.to_numeric(
            df[numeric_col],
            errors="coerce",
        )

query_display_col = first_column(
    df,
    "Query_Entity",
    "Query_Entity_Normalized",
)

query_norm_col = first_column(
    df,
    "Query_Entity_Normalized",
    "Query_Entity",
)

predicted_col = first_column(
    df,
    "Predicted_Entity",
    "Target",
)

predicted_type_col = first_column(
    df,
    "Predicted_Entity_Type",
    "Target_Type",
)

query_type_col = first_column(
    df,
    "Query_Entity_Type",
    "Source_Type",
)

class_col = first_column(
    df,
    "Hypothesis_Class",
    "Hypothesis_Type",
)

score_col = first_column(
    df,
    "AI_Score",
    "Final_AI_Score",
    "Score",
)

validation_col = first_column(
    df,
    "Validation_Status",
    "Temporal_Validation_Status",
)

bridge_col = first_column(
    df,
    "Bridge_Nodes",
    "Bridge_Nodes_Display",
)

node2vec_col = first_column(
    df,
    "Node2Vec_Integrated_Score",
    "Embedding_Integrated_Score",
    "Node2Vec_Score",
)

ml_col = first_column(
    df,
    "ML_Probability",
    "Machine_Learning_Probability",
)

rank_col = first_column(
    df,
    "Rank_For_Query",
    "Rank",
)

required = [
    query_display_col,
    query_norm_col,
    predicted_col,
    score_col,
]

if any(column is None for column in required):
    st.error(
        "The searchable database is missing required query columns."
    )
    st.write("Detected columns:", list(df.columns))
    st.stop()


# ============================================================
# HEADER
# ============================================================

hero(
    "🔍 Search STX-LBD Hypotheses",
    "Search a biological entity and refine its predicted relationships by "
    "hypothesis class, entity type, temporal validation status, and AI score. "
    "Validated predictions can be traced to their supporting post-2015 literature.",
)


# ============================================================
# SEARCH CONTROLS
# ============================================================

entity_table = (
    df[
        [
            query_display_col,
            query_norm_col,
        ]
    ]
    .dropna()
    .drop_duplicates()
    .copy()
)

entity_table[query_display_col] = (
    entity_table[query_display_col]
    .astype(str)
    .str.strip()
)

entity_table[query_norm_col] = (
    entity_table[query_norm_col]
    .astype(str)
    .str.strip()
    .str.casefold()
)

entity_table = entity_table[
    entity_table[query_display_col].ne("")
].sort_values(query_display_col)

display_to_normalized = dict(
    zip(
        entity_table[query_display_col],
        entity_table[query_norm_col],
    )
)

entity_options = list(display_to_normalized)

selected_entity = st.selectbox(
    "Biological entity",
    options=[""] + entity_options,
    index=0,
    placeholder=(
        "Type or select: sxtA, temperature, salinity, Alexandrium..."
    ),
)

with st.expander(
    "Refine results",
    expanded=True,
):
    f1, f2, f3 = st.columns(3)

    with f1:
        classes = (
            sorted(
                df[class_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
            if class_col
            else []
        )

        selected_classes = st.multiselect(
            "Hypothesis class",
            options=classes,
            placeholder="All classes",
        )

    with f2:
        entity_types = (
            sorted(
                df[predicted_type_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
            if predicted_type_col
            else []
        )

        selected_types = st.multiselect(
            "Predicted entity type",
            options=entity_types,
            placeholder="All entity types",
        )

    with f3:
        validation_states = (
            sorted(
                df[validation_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
            if validation_col
            else []
        )

        selected_validation = st.multiselect(
            "Temporal validation",
            options=validation_states,
            placeholder="All states",
        )

    f4, f5 = st.columns(2)

    with f4:
        scores = df[score_col].dropna()

        score_min = (
            float(scores.min())
            if not scores.empty
            else 0.0
        )

        score_max = (
            float(scores.max())
            if not scores.empty
            else 1.0
        )

        minimum_score = st.slider(
            "Minimum AI score",
            min_value=score_min,
            max_value=score_max,
            value=score_min,
            step=0.01,
            format="%.2f",
        )

    with f5:
        result_limit = st.slider(
            "Maximum number of results",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )

search_clicked = st.button(
    "Search STX-LBD",
    use_container_width=True,
    type="primary",
)

if not search_clicked:
    st.info(
        "Select a biological entity, adjust the optional filters, "
        "and click **Search STX-LBD**."
    )
    st.stop()

if not selected_entity:
    st.warning(
        "Select a biological entity before searching."
    )
    st.stop()


# ============================================================
# SEARCH
# ============================================================

normalized_entity = display_to_normalized[
    selected_entity
]

results = df[
    df[query_norm_col]
    .astype(str)
    .str.strip()
    .str.casefold()
    .eq(normalized_entity)
].copy()

if selected_classes and class_col:
    results = results[
        results[class_col]
        .astype(str)
        .isin(selected_classes)
    ]

if selected_types and predicted_type_col:
    results = results[
        results[predicted_type_col]
        .astype(str)
        .isin(selected_types)
    ]

if selected_validation and validation_col:
    results = results[
        results[validation_col]
        .astype(str)
        .isin(selected_validation)
    ]

results = results[
    results[score_col].fillna(-1)
    >= minimum_score
]

results = (
    results
    .sort_values(
        score_col,
        ascending=False,
    )
    .head(result_limit)
    .reset_index(drop=True)
)

if results.empty:
    st.info(
        "No hypotheses matched the selected entity and filters."
    )
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

section(
    f"Results for {selected_entity}",
    f"{len(results)} ranked relationship(s) displayed.",
)

overview_cols = [
    column
    for column in [
        rank_col,
        predicted_col,
        predicted_type_col,
        class_col,
        score_col,
        validation_col,
    ]
    if column
]

overview = results[
    overview_cols
].copy()

if score_col in overview.columns:
    overview[score_col] = (
        overview[score_col]
        .round(3)
    )

st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True,
    column_config={
        score_col: st.column_config.NumberColumn(
            "AI score",
            format="%.3f",
        )
    }
    if score_col
    else None,
)


# ============================================================
# DETAILED RESULT CARDS
# ============================================================

section(
    "Biological interpretation",
    "Each prediction includes model evidence and, where available, "
    "traceable post-2015 literature used for temporal validation.",
)

for display_rank, (_, row) in enumerate(
    results.iterrows(),
    start=1,
):
    predicted = text_value(
        row.get(predicted_col)
    )

    validation = (
        text_value(
            row.get(validation_col),
            "Not assessed",
        )
        if validation_col
        else "Not assessed"
    )

    hypothesis_class = (
        text_value(
            row.get(class_col)
        )
        if class_col
        else "Not available"
    )

    predicted_type = (
        text_value(
            row.get(predicted_type_col)
        )
        if predicted_type_col
        else "Not available"
    )

    bridge_nodes = (
        text_value(
            row.get(bridge_col)
        )
        if bridge_col
        else "Not available"
    )

    query_type = (
        row.get(
            query_type_col,
            "UNKNOWN",
        )
        if query_type_col
        else "UNKNOWN"
    )

    interpretation = generate_biological_interpretation(
        entity_a=selected_entity,
        type_a=query_type,
        entity_b=predicted,
        type_b=(
            row.get(
                predicted_type_col,
                "UNKNOWN",
            )
            if predicted_type_col
            else "UNKNOWN"
        ),
        hypothesis_class=(
            row.get(
                class_col,
                "",
            )
            if class_col
            else ""
        ),
        bridge_nodes=(
            row.get(
                bridge_col,
                "",
            )
            if bridge_col
            else ""
        ),
    )

    literature_evidence = (
        get_hypothesis_evidence(
            selected_entity,
            predicted,
        )
        if validation.casefold() == "validated"
        else pd.DataFrame()
    )

    with st.container(border=True):
        st.markdown(
            f"### {display_rank}. "
            f"{selected_entity} ↔ {predicted}"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "AI score",
                score_value(
                    row.get(score_col)
                ),
            )

        with m2:
            st.metric(
                "Temporal validation",
                validation,
            )

        with m3:
            st.metric(
                "Hypothesis class",
                hypothesis_class,
            )

        with m4:
            st.metric(
                "Predicted entity type",
                predicted_type,
            )

        st.markdown(
            "#### Biological interpretation"
        )

        st.info(
            interpretation
        )

        # ----------------------------------------------------
        # Supporting model evidence
        # ----------------------------------------------------

        with st.expander(
            "Supporting graph and model evidence"
        ):
            st.markdown(
                "**Bridge nodes**"
            )

            st.write(
                bridge_nodes
            )

            e1, e2 = st.columns(2)

            with e1:
                if node2vec_col:
                    st.metric(
                        "Node2Vec score",
                        score_value(
                            row.get(node2vec_col)
                        ),
                    )

            with e2:
                if ml_col:
                    st.metric(
                        "Machine-learning probability",
                        score_value(
                            row.get(ml_col)
                        ),
                    )

        # ----------------------------------------------------
        # Temporal validation literature
        # ----------------------------------------------------

        st.markdown(
            "#### Temporal validation evidence"
        )

        if validation.casefold() == "validated":

            if evidence_df is None:
                st.warning(
                    "The hypothesis is temporally validated, but the "
                    "paper-level evidence table has not been loaded. "
                    f"Expected file: `{evidence_path}`"
                )

            elif literature_evidence.empty:
                st.warning(
                    "This relationship is labeled as validated, but no "
                    "matching paper-level provenance was found in "
                    "`hypothesis_validation_evidence.csv`."
                )

            else:
                publication_count = len(
                    literature_evidence
                )

                valid_years = (
                    literature_evidence["Year"]
                    .dropna()
                    if "Year"
                    in literature_evidence.columns
                    else pd.Series(
                        dtype=float
                    )
                )

                first_year = (
                    int(valid_years.min())
                    if not valid_years.empty
                    else "Not available"
                )

                last_year = (
                    int(valid_years.max())
                    if not valid_years.empty
                    else "Not available"
                )

                t1, t2, t3, t4 = st.columns(4)

                with t1:
                    st.metric(
                        "Supporting publications",
                        publication_count,
                    )

                with t2:
                    st.metric(
                        "First observed",
                        first_year,
                    )

                with t3:
                    st.metric(
                        "Most recent evidence",
                        last_year,
                    )

                with t4:
                    st.metric(
                        "Literature support",
                        evidence_strength(
                            publication_count
                        ),
                    )

                st.success(
                    "This predicted semantic relationship subsequently "
                    "appeared in the independent post-2015 literature. "
                    "The publications below provide the traceable "
                    "literature provenance for that temporal validation."
                )

                # Default: show five most recent papers.
                recent_evidence = (
                    literature_evidence.head(5)
                )

                with st.expander(
                    "View supporting post-2015 literature",
                    expanded=False,
                ):
                    for paper_number, (
                        _,
                        paper_row,
                    ) in enumerate(
                        recent_evidence.iterrows(),
                        start=1,
                    ):
                        render_publication(
                            paper_row,
                            paper_number,
                        )

                        if (
                            paper_number
                            < len(recent_evidence)
                        ):
                            st.divider()

                    remaining = (
                        publication_count
                        - len(
                            recent_evidence
                        )
                    )

                    if remaining > 0:
                        st.caption(
                            f"{remaining} additional supporting "
                            "publication(s) are available in the "
                            "downloadable evidence table."
                        )

                evidence_filename = (
                    f"{selected_entity}_"
                    f"{predicted}_"
                    "validation_evidence.csv"
                )

                safe_evidence_filename = (
                    evidence_filename
                    .replace(
                        " ",
                        "_",
                    )
                    .replace(
                        "/",
                        "_",
                    )
                    .replace(
                        "\\",
                        "_",
                    )
                )

                st.download_button(
                    "Download temporal validation evidence",
                    data=(
                        literature_evidence
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        )
                    ),
                    file_name=(
                        safe_evidence_filename
                    ),
                    mime="text/csv",
                    key=(
                        f"download_evidence_"
                        f"{display_rank}"
                    ),
                )

        elif validation.casefold() == "unvalidated":
            st.warning(
                "No supporting post-2015 semantic edge was identified "
                "during temporal validation. This relationship therefore "
                "remains a candidate hypothesis rather than a "
                "literature-supported future-positive relationship."
            )

        else:
            st.info(
                "A temporal validation label is not available for "
                "this relationship."
            )


# ============================================================
# DOWNLOAD SEARCH RESULTS
# ============================================================

results["Interpretation"] = results.apply(
    lambda row: generate_biological_interpretation(
        entity_a=selected_entity,
        type_a=(
            row.get(
                query_type_col,
                "UNKNOWN",
            )
            if query_type_col
            else "UNKNOWN"
        ),
        entity_b=row.get(
            predicted_col,
            "",
        ),
        type_b=(
            row.get(
                predicted_type_col,
                "UNKNOWN",
            )
            if predicted_type_col
            else "UNKNOWN"
        ),
        hypothesis_class=(
            row.get(
                class_col,
                "",
            )
            if class_col
            else ""
        ),
        bridge_nodes=(
            row.get(
                bridge_col,
                "",
            )
            if bridge_col
            else ""
        ),
    ),
    axis=1,
)

safe_entity = (
    selected_entity
    .replace(
        " ",
        "_",
    )
    .replace(
        "/",
        "_",
    )
    .replace(
        "\\",
        "_",
    )
)

st.download_button(
    "Download filtered hypothesis results as CSV",
    results.to_csv(
        index=False
    ).encode(
        "utf-8"
    ),
    file_name=(
        f"{safe_entity}_STX_LBD_hypotheses.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)
