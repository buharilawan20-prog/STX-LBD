
import sys
from pathlib import Path
import streamlit as st

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import apply_global_style, hero, render_sidebar

st.set_page_config(
    page_title="STX-LBD Documentation",
    page_icon="📖",
    layout="wide",
)

apply_global_style()
render_sidebar()

hero(
    "📖 STX-LBD Documentation",
    "Methods, interpretation guidance, data and code availability, and citation information.",
)

with st.expander("Overview", expanded=True):
    st.markdown(
        """
        STX-LBD is an artificial intelligence-guided literature-based discovery
        framework for marine saxitoxin research. It transforms fragmented
        scientific literature into structured biological knowledge for hypothesis
        generation, prioritization, temporal validation, and interactive exploration.
        """
    )

with st.expander("Core workflow", expanded=True):
    st.markdown(
        """
        1. Multi-database literature collection and corpus enrichment  
        2. Semantic entity extraction and normalization  
        3. Semantic knowledge graph construction  
        4. Node2Vec graph representation learning  
        5. Supervised machine-learning hypothesis ranking  
        6. Strict temporal validation using post-2015 literature  
        7. Cross-taxa knowledge transfer between cyanobacteria and dinoflagellates  
        8. Interactive hypothesis and knowledge-graph exploration  
        """
    )

with st.expander("Knowledge graph explorer"):
    st.markdown(
        """
        The knowledge-graph explorer supports:

        - one- and two-hop semantic neighborhoods;
        - entity-type filtering;
        - minimum semantic-support filtering;
        - interactive network navigation;
        - node profiles and ranked neighbors;
        - linked AI-ranked hypotheses;
        - CSV, GraphML, GEXF, and JSON downloads.
        """
    )

with st.expander("How to interpret predictions"):
    st.markdown(
        """
        **Validated** means that a predicted semantic relationship subsequently
        appeared in the independent post-2015 knowledge graph. It does not
        necessarily mean that the relationship has been established experimentally
        as a causal mechanism.

        **Unvalidated** means that the relationship was not observed in the
        post-2015 validation graph and therefore remains a candidate hypothesis.
        """
    )

with st.expander("Data and code availability"):
    st.markdown(
        """
        **GitHub repository**  
        https://github.com/buharilawan20-prog/STX-LBD

        **Zenodo DOI**  
        https://doi.org/10.5281/zenodo.21640517
        """
    )

with st.expander("Citation"):
    st.markdown(
        """
        *STX-LBD: An AI-Guided Literature-Based Discovery Framework for
        Predicting Future Discoveries in Marine Saxitoxin Research*
        """
    )

st.warning(
    "STX-LBD is a hypothesis-generation and prioritization resource. "
    "Predictions should be independently evaluated before being treated as "
    "experimentally established biological mechanisms."
)
