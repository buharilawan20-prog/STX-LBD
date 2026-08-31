
from __future__ import annotations

from pathlib import Path
import re
import sys

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

from ui import apply_global_style, hero, render_sidebar, section


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="STX-LBD Discovery Analytics",
    page_icon="📊",
    layout="wide",
)

apply_global_style()
render_sidebar()


# ============================================================
# FILE DISCOVERY
# ============================================================

FINAL_WORKSPACE = PROJECT_ROOT / "FINAL_WORKSPACE"
KG_DIR = FINAL_WORKSPACE / "kg"
SPLIT_DIR = FINAL_WORKSPACE / "splits"
POC_DIR = FINAL_WORKSPACE / "proof_of_concept"


def locate_exact_or_pattern(
    base_dir: Path,
    exact_name: str,
    patterns: list[str],
) -> Path | None:
    exact = base_dir / exact_name
    if exact.exists():
        return exact

    if base_dir.exists():
        for pattern in patterns:
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return matches[0]

    if FINAL_WORKSPACE.exists():
        exact_matches = sorted(FINAL_WORKSPACE.rglob(exact_name))
        if exact_matches:
            return exact_matches[0]

        for pattern in patterns:
            matches = sorted(FINAL_WORKSPACE.rglob(pattern))
            if matches:
                return matches[0]

    return None


FILES = {
    "searchable_hypotheses": POC_DIR / "searchable_hypotheses.csv",
    "validation_evidence": POC_DIR / "hypothesis_validation_evidence.csv",
    "dino_all_edges": locate_exact_or_pattern(
        KG_DIR,
        "dino_all_semantic_edges.csv",
        ["*dino*all*semantic*edge*.csv"],
    ),
    "cyano_all_edges": locate_exact_or_pattern(
        KG_DIR,
        "cyano_all_semantic_edges.csv",
        ["*cyano*all*semantic*edge*.csv"],
    ),
    "dino_pre_edges": locate_exact_or_pattern(
        KG_DIR,
        "dino_pre2016_semantic_edges.csv",
        ["*dino*pre*2016*semantic*edge*.csv"],
    ),
    "dino_post_edges": locate_exact_or_pattern(
        KG_DIR,
        "dino_post2015_semantic_edges.csv",
        ["*dino*post*2015*semantic*edge*.csv"],
    ),
    "dino_pre_docs": locate_exact_or_pattern(
        SPLIT_DIR,
        "dino_pre2016.csv",
        ["*dino*pre*2016*.csv"],
    ),
    "dino_post_docs": locate_exact_or_pattern(
        SPLIT_DIR,
        "dino_post2015.csv",
        ["*dino*post*2015*.csv"],
    ),
    "cyano_docs": locate_exact_or_pattern(
        SPLIT_DIR,
        "cyano_all.csv",
        ["*cyano*all*.csv", "*cyano*.csv"],
    ),
}


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def read_csv(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def safe_read(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None

    try:
        return read_csv(str(path))
    except Exception:
        return None


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


def pair_key(source: object, target: object) -> str:
    return "||".join(
        sorted(
            [
                canonical(source),
                canonical(target),
            ]
        )
    )


def count_rows(frame: pd.DataFrame | None) -> int | None:
    return None if frame is None else len(frame)


def numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()


# ============================================================
# LOAD AVAILABLE DATA
# ============================================================

hypotheses = safe_read(FILES["searchable_hypotheses"])
validation_evidence = safe_read(FILES["validation_evidence"])
dino_all = safe_read(FILES["dino_all_edges"])
cyano_all = safe_read(FILES["cyano_all_edges"])
dino_pre = safe_read(FILES["dino_pre_edges"])
dino_post = safe_read(FILES["dino_post_edges"])
dino_pre_docs = safe_read(FILES["dino_pre_docs"])
dino_post_docs = safe_read(FILES["dino_post_docs"])
cyano_docs = safe_read(FILES["cyano_docs"])


# ============================================================
# HEADER
# ============================================================

hero(
    "📊 STX-LBD Discovery Analytics",
    "Explore the corpus, knowledge graphs, hypothesis landscape, temporal validation, "
    "model performance, and cross-taxa discovery signals underlying STX-LBD.",
)

st.caption(
    "Most panels are computed directly from the current STX-LBD output files. "
    "Model-performance panels reproduce the benchmark values reported for the current study."
)


# ============================================================
# TOP-LEVEL SUMMARY
# ============================================================

section(
    "System summary",
    "Current discovery outputs and validation resources.",
)

summary_values = []

publication_total = None
if dino_pre_docs is not None and dino_post_docs is not None:
    # This is the dinoflagellate temporal corpus, not necessarily the full 1,749 corpus.
    dino_temporal_total = len(dino_pre_docs) + len(dino_post_docs)
else:
    dino_temporal_total = None

priority_hypotheses = len(hypotheses) if hypotheses is not None else None

validated_count = None
if hypotheses is not None:
    validation_col = find_col(
        hypotheses,
        "Validation_Status",
        "Temporal_Validation_Status",
    )
    if validation_col:
        validated_count = int(
            hypotheses[validation_col]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("validated")
            .sum()
        )

unique_supporting_papers = None
if validation_evidence is not None:
    doc_col = find_col(
        validation_evidence,
        "Document_ID",
        "document_id",
    )
    if doc_col:
        unique_supporting_papers = (
            validation_evidence[doc_col]
            .replace("", pd.NA)
            .dropna()
            .astype(str)
            .nunique()
        )

top_metrics = [
    ("1,749", "Enriched publications"),
    (
        f"{priority_hypotheses:,}"
        if priority_hypotheses is not None
        else "514",
        "Priority hypotheses",
    ),
    (
        f"{validated_count:,}"
        if validated_count is not None
        else "135",
        "Temporally validated hypotheses",
    ),
    (
        f"{unique_supporting_papers:,}"
        if unique_supporting_papers is not None
        else "—",
        "Traceable validation papers",
    ),
]

for column, (value, label) in zip(st.columns(4), top_metrics):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TABS
# ============================================================

tab_corpus, tab_hypotheses, tab_temporal, tab_cross = st.tabs(
    [
        "Corpus & Knowledge Graphs",
        "Hypothesis Discovery",
        "Temporal Validation",
        "Cross-Taxa Analytics",
    ]
)


# ============================================================
# TAB 1 — CORPUS & KGs
# ============================================================

with tab_corpus:

    section(
        "Temporal and taxonomic corpus structure",
        "Document counts used for dinoflagellate temporal validation and cyanobacterial comparison.",
    )

    corpus_rows = []

    if dino_pre_docs is not None:
        corpus_rows.append(
            {
                "Corpus": "Dinoflagellate pre-2016",
                "Records": len(dino_pre_docs),
            }
        )

    if dino_post_docs is not None:
        corpus_rows.append(
            {
                "Corpus": "Dinoflagellate post-2015",
                "Records": len(dino_post_docs),
            }
        )

    if cyano_docs is not None:
        corpus_rows.append(
            {
                "Corpus": "Cyanobacteria",
                "Records": len(cyano_docs),
            }
        )

    if corpus_rows:
        corpus_df = pd.DataFrame(corpus_rows)
        st.bar_chart(
            corpus_df.set_index("Corpus"),
            horizontal=True,
        )
        st.dataframe(
            corpus_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Temporal split CSVs were not detected automatically. "
            "The graph and hypothesis analytics below remain available."
        )

    section(
        "Knowledge graph sizes",
        "Semantic edge counts loaded directly from the final graph files.",
    )

    kg_rows = []

    for label, frame in [
        ("Dinoflagellate all", dino_all),
        ("Dinoflagellate pre-2016", dino_pre),
        ("Dinoflagellate post-2015", dino_post),
        ("Cyanobacteria all", cyano_all),
    ]:
        if frame is None:
            continue

        source_col = find_col(frame, "source", "Source")
        target_col = find_col(frame, "target", "Target")

        if source_col and target_col:
            nodes = (
                set(frame[source_col].dropna().astype(str))
                | set(frame[target_col].dropna().astype(str))
            )
            node_count = len(nodes)
        else:
            node_count = None

        kg_rows.append(
            {
                "Knowledge graph": label,
                "Nodes": node_count,
                "Edges": len(frame),
            }
        )

    if kg_rows:
        kg_df = pd.DataFrame(kg_rows)

        k1, k2 = st.columns(2)

        with k1:
            st.markdown("#### Semantic edges")
            st.bar_chart(
                kg_df.set_index("Knowledge graph")["Edges"],
                horizontal=True,
            )

        with k2:
            st.markdown("#### Nodes")
            if kg_df["Nodes"].notna().any():
                st.bar_chart(
                    kg_df.dropna(subset=["Nodes"])
                    .set_index("Knowledge graph")["Nodes"],
                    horizontal=True,
                )

        st.dataframe(
            kg_df,
            use_container_width=True,
            hide_index=True,
        )

    # Entity-type composition from dino_all if type columns exist.
    if dino_all is not None:
        source_type_col = find_col(dino_all, "source_type", "Source_Type")
        target_type_col = find_col(dino_all, "target_type", "Target_Type")

        if source_type_col and target_type_col:
            entity_types = pd.concat(
                [
                    dino_all[source_type_col],
                    dino_all[target_type_col],
                ],
                ignore_index=True,
            )
            entity_counts = (
                entity_types
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .rename_axis("Entity type")
                .reset_index(name="Occurrences in edges")
            )

            section(
                "Entity-type representation in the dinoflagellate graph",
                "Counts reflect appearances of entity types across semantic edges.",
            )

            st.bar_chart(
                entity_counts.set_index("Entity type"),
                horizontal=True,
            )


# ============================================================
# TAB 2 — HYPOTHESIS DISCOVERY
# ============================================================

with tab_hypotheses:

    if hypotheses is None:
        st.warning(
            "`searchable_hypotheses.csv` was not found, so hypothesis analytics cannot be displayed."
        )

    else:
        class_col = find_col(
            hypotheses,
            "Hypothesis_Class",
            "Hypothesis_Type",
        )

        score_col = find_col(
            hypotheses,
            "AI_Score",
            "Final_AI_Score",
            "Score",
        )

        validation_col = find_col(
            hypotheses,
            "Validation_Status",
            "Temporal_Validation_Status",
        )

        section(
            "Hypothesis-class composition",
            "Distribution of the current searchable/prioritized hypothesis set.",
        )

        if class_col:
            class_counts = (
                hypotheses[class_col]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .rename_axis("Hypothesis class")
                .reset_index(name="Hypotheses")
            )

            st.bar_chart(
                class_counts.set_index("Hypothesis class"),
                horizontal=True,
            )

            st.dataframe(
                class_counts,
                use_container_width=True,
                hide_index=True,
            )

        if validation_col:
            section(
                "Validated vs unvalidated hypotheses",
                "Temporal validation status in the searchable hypothesis database.",
            )

            validation_counts = (
                hypotheses[validation_col]
                .fillna("Not assessed")
                .astype(str)
                .str.strip()
                .value_counts()
                .rename_axis("Validation status")
                .reset_index(name="Hypotheses")
            )

            v1, v2 = st.columns(2)

            with v1:
                st.bar_chart(
                    validation_counts.set_index("Validation status"),
                    horizontal=True,
                )

            with v2:
                st.dataframe(
                    validation_counts,
                    use_container_width=True,
                    hide_index=True,
                )

        if score_col:
            scores = pd.to_numeric(
                hypotheses[score_col],
                errors="coerce",
            ).dropna()

            if not scores.empty:
                section(
                    "AI-score distribution",
                    "Distribution of final AI ranking scores across the searchable hypothesis set.",
                )

                bins = pd.cut(
                    scores,
                    bins=10,
                    include_lowest=True,
                )

                score_hist = (
                    bins.value_counts()
                    .sort_index()
                    .rename_axis("AI-score interval")
                    .reset_index(name="Hypotheses")
                )

                score_hist["AI-score interval"] = (
                    score_hist["AI-score interval"].astype(str)
                )

                st.bar_chart(
                    score_hist.set_index("AI-score interval")
                )

                q1, q2, q3, q4 = st.columns(4)

                q1.metric(
                    "Mean AI score",
                    f"{scores.mean():.3f}",
                )
                q2.metric(
                    "Median AI score",
                    f"{scores.median():.3f}",
                )
                q3.metric(
                    "Maximum AI score",
                    f"{scores.max():.3f}",
                )
                q4.metric(
                    "Minimum AI score",
                    f"{scores.min():.3f}",
                )


# ============================================================
# TAB 3 — TEMPORAL VALIDATION
# ============================================================

with tab_temporal:

    section(
        "AI ranking vs unsupervised Node2Vec",
        "Strict temporal validation performance reported for the enriched STX-LBD hypothesis set.",
    )

    performance_df = pd.DataFrame(
        {
            "K": [10, 20, 50, 100, 200],
            "Node2Vec Precision@K": [0.20, 0.40, 0.50, 0.45, 0.375],
            "AI Precision@K": [1.00, 0.95, 0.92, 0.74, 0.60],
            "Node2Vec Hits@K": [2, 8, 25, 45, 75],
            "AI Hits@K": [10, 19, 46, 74, 120],
        }
    )

    chart_df = performance_df.set_index("K")[
        [
            "Node2Vec Precision@K",
            "AI Precision@K",
        ]
    ]

    st.line_chart(chart_df)

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True,
    )

    section(
        "Machine-learning benchmark",
        "Model discrimination metrics from the strict temporal ranking evaluation.",
    )

    model_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
                "Extra Trees",
                "SVM",
                "MLP",
            ],
            "ROC_AUC": [
                0.710081,
                0.758427,
                0.741476,
                0.750738,
                0.643963,
                0.724205,
            ],
            "PR_AUC": [
                0.469861,
                0.425250,
                0.449985,
                0.430156,
                0.407302,
                0.380385,
            ],
            "RR": [
                1.0,
                0.5,
                1.0,
                1.0,
                1.0,
                0.5,
            ],
        }
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
    )

    best_roc = model_df.loc[
        model_df["ROC_AUC"].idxmax()
    ]

    best_pr = model_df.loc[
        model_df["PR_AUC"].idxmax()
    ]

    b1, b2 = st.columns(2)

    b1.metric(
        "Best ROC-AUC",
        f"{best_roc['ROC_AUC']:.3f}",
        help=str(best_roc["Model"]),
    )

    b2.metric(
        "Best PR-AUC",
        f"{best_pr['PR_AUC']:.3f}",
        help=str(best_pr["Model"]),
    )

    if validation_evidence is not None:
        section(
            "Traceable post-2015 validation evidence",
            "Paper-level provenance linked to temporally validated hypotheses.",
        )

        query_col = find_col(
            validation_evidence,
            "Query_Entity",
        )
        predicted_col = find_col(
            validation_evidence,
            "Predicted_Entity",
        )
        doc_col = find_col(
            validation_evidence,
            "Document_ID",
        )
        year_col = find_col(
            validation_evidence,
            "Year",
        )
        match_col = find_col(
            validation_evidence,
            "Evidence_Match",
        )

        evidence = validation_evidence.copy()

        if match_col:
            evidence = evidence[
                evidence[match_col]
                .astype(str)
                .str.strip()
                .str.casefold()
                .eq("matched")
            ]

        unique_hypotheses = (
            evidence[
                [query_col, predicted_col]
            ]
            .drop_duplicates()
            .shape[0]
            if query_col and predicted_col
            else None
        )

        unique_docs = (
            evidence[doc_col]
            .replace("", pd.NA)
            .dropna()
            .astype(str)
            .nunique()
            if doc_col
            else None
        )

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Validated hypotheses with provenance",
            f"{unique_hypotheses:,}"
            if unique_hypotheses is not None
            else "—",
        )

        e2.metric(
            "Unique supporting publications",
            f"{unique_docs:,}"
            if unique_docs is not None
            else "—",
        )

        e3.metric(
            "Paper-level evidence rows",
            f"{len(evidence):,}",
        )

        if year_col:
            years = pd.to_numeric(
                evidence[year_col],
                errors="coerce",
            ).dropna()

            if not years.empty:
                year_counts = (
                    years.astype(int)
                    .value_counts()
                    .sort_index()
                    .rename_axis("Year")
                    .reset_index(name="Evidence rows")
                )

                st.bar_chart(
                    year_counts.set_index("Year")
                )


# ============================================================
# TAB 4 — CROSS-TAXA ANALYTICS
# ============================================================

with tab_cross:

    if dino_all is None or cyano_all is None:
        st.warning(
            "Both final dinoflagellate and cyanobacterial graph files are required for cross-taxa analytics."
        )

    else:
        d_source = find_col(dino_all, "source", "Source")
        d_target = find_col(dino_all, "target", "Target")
        c_source = find_col(cyano_all, "source", "Source")
        c_target = find_col(cyano_all, "target", "Target")

        dino_pair_set = {
            pair_key(source, target)
            for source, target in zip(
                dino_all[d_source],
                dino_all[d_target],
            )
        }

        cyano_pair_set = {
            pair_key(source, target)
            for source, target in zip(
                cyano_all[c_source],
                cyano_all[c_target],
            )
        }

        shared = dino_pair_set & cyano_pair_set
        dino_only = dino_pair_set - cyano_pair_set
        cyano_only = cyano_pair_set - dino_pair_set

        section(
            "Shared and lineage-specific semantic knowledge",
            "Direct comparison of final dinoflagellate and cyanobacterial semantic edge sets.",
        )

        cross_df = pd.DataFrame(
            {
                "Category": [
                    "Shared across taxa",
                    "Dinoflagellate-specific",
                    "Cyanobacteria-specific",
                ],
                "Relationships": [
                    len(shared),
                    len(dino_only),
                    len(cyano_only),
                ],
            }
        )

        st.bar_chart(
            cross_df.set_index("Category"),
            horizontal=True,
        )

        st.dataframe(
            cross_df,
            use_container_width=True,
            hide_index=True,
        )

        if dino_pre is not None and dino_post is not None:
            pre_source = find_col(dino_pre, "source", "Source")
            pre_target = find_col(dino_pre, "target", "Target")
            post_source = find_col(dino_post, "source", "Source")
            post_target = find_col(dino_post, "target", "Target")

            pre_pairs = {
                pair_key(source, target)
                for source, target in zip(
                    dino_pre[pre_source],
                    dino_pre[pre_target],
                )
            }

            post_pairs = {
                pair_key(source, target)
                for source, target in zip(
                    dino_post[post_source],
                    dino_post[post_target],
                )
            }

            cyano_absent_pre = cyano_pair_set - pre_pairs
            later_supported = cyano_absent_pre & post_pairs
            still_cyano_only = cyano_absent_pre - post_pairs

            section(
                "Cross-taxa transfer/convergence signals",
                "Cyanobacterial relationships absent from the pre-2016 dinoflagellate graph and their later representation in post-2015 dinoflagellate literature.",
            )

            transfer_df = pd.DataFrame(
                {
                    "Signal": [
                        "Cyano signals absent from pre-2016 dino KG",
                        "Later represented post-2015",
                        "Still cyano-only",
                    ],
                    "Relationships": [
                        len(cyano_absent_pre),
                        len(later_supported),
                        len(still_cyano_only),
                    ],
                }
            )

            st.bar_chart(
                transfer_df.set_index("Signal"),
                horizontal=True,
            )

            st.dataframe(
                transfer_df,
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            "Cross-taxa overlap is interpreted as comparative semantic evidence. "
            "It does not independently establish horizontal gene transfer, common ancestry, "
            "or identical mechanisms across lineages."
        )
