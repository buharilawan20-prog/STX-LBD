from pathlib import Path
import sys

import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from query_engine import STXLBD


st.set_page_config(
    page_title="STX-LBD Explorer",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_engine():
    return STXLBD()


engine = load_engine()

st.title("STX-LBD Explorer")

st.write(
    "Search AI-ranked hypotheses involving saxitoxin biosynthesis genes, "
    "taxa, toxins, environmental factors, and biological processes."
)

entity = st.text_input(
    "Search for an entity",
    placeholder="Examples: sxtA, temperature, Alexandrium",
)

col1, col2 = st.columns(2)

with col1:
    top_n = st.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

with col2:
    validated_only = st.checkbox(
        "Show validated hypotheses only"
    )

search_clicked = st.button(
    "Search",
    type="primary",
)

if search_clicked:

    if not entity.strip():
        st.warning("Enter an entity before searching.")
        st.stop()

    results = engine.search(
        entity=entity,
        top_n=top_n,
        validated_only=validated_only,
    )

    if results.empty:
        st.warning(
            f"No hypotheses were found for '{entity}'. "
            "Check the spelling or try another normalized entity."
        )
        st.stop()

    st.subheader(f"Results for: {entity}")

    display_columns = [
        "Rank_For_Query",
        "Predicted_Entity",
        "Predicted_Entity_Type",
        "Hypothesis_Class",
        "AI_Score",
        "Validation_Status",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detailed hypotheses")

    for _, row in results.iterrows():

        predicted_entity = row.get(
            "Predicted_Entity",
            "Unknown",
        )

        ai_score = row.get(
            "AI_Score",
            float("nan"),
        )

        validation_status = row.get(
            "Validation_Status",
            "Not assessed",
        )

        hypothesis_class = row.get(
            "Hypothesis_Class",
            "Unclassified",
        )

        title = (
            f"{row.get('Rank_For_Query', '-')}. "
            f"{entity} ↔ {predicted_entity}"
        )

        with st.expander(title):

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                if pd.notna(ai_score):
                    st.metric(
                        "AI score",
                        f"{float(ai_score):.3f}",
                    )
                else:
                    st.metric(
                        "AI score",
                        "Not available",
                    )

            with metric2:
                st.metric(
                    "Validation",
                    validation_status,
                )

            with metric3:
                st.metric(
                    "Hypothesis class",
                    hypothesis_class,
                )

            st.markdown("**Predicted relationship**")

            st.write(
                f"{entity} ↔ {predicted_entity}"
            )

            bridge_nodes = row.get(
                "Bridge_Nodes",
                "",
            )

            if pd.notna(bridge_nodes) and str(
                bridge_nodes
            ).strip():
                st.markdown("**Bridge nodes**")
                st.write(bridge_nodes)

            interpretation = row.get(
                "Interpretation",
                "",
            )

            if pd.notna(interpretation) and str(
                interpretation
            ).strip():
                st.markdown("**Interpretation**")
                st.write(interpretation)

            node2vec_score = row.get(
                "Node2Vec_Integrated_Score",
                None,
            )

            ml_probability = row.get(
                "ML_Probability",
                None,
            )

            score_col1, score_col2 = st.columns(2)

            with score_col1:
                if (
                    node2vec_score is not None
                    and pd.notna(node2vec_score)
                ):
                    st.write(
                        "Node2Vec score:",
                        round(float(node2vec_score), 3),
                    )

            with score_col2:
                if (
                    ml_probability is not None
                    and pd.notna(ml_probability)
                ):
                    st.write(
                        "ML probability:",
                        round(float(ml_probability), 3),
                    )

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    safe_entity = (
        entity.strip()
        .replace(" ", "_")
        .replace("/", "_")
    )

    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name=f"{safe_entity}_stx_lbd_results.csv",
        mime="text/csv",
    )
