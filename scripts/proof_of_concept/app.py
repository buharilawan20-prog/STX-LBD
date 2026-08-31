
import streamlit as st

from ui import apply_global_style, hero, render_sidebar, section


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="STX-LBD Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_style()
render_sidebar()


# ============================================================
# HERO
# ============================================================

hero(
    "STX-LBD Explorer",
    "An AI-guided literature-based discovery platform for exploring predicted "
    "biological relationships in marine saxitoxin research. Search AI-ranked "
    "hypotheses, inspect temporal validation evidence, explore semantic knowledge "
    "graphs, and compare STX biology across dinoflagellates and cyanobacteria.",
    home=True,
)

st.caption(
    "STX-LBD Explorer · Research software accompanying the STX-LBD framework"
)


# ============================================================
# PLATFORM MODULES
# ============================================================

section(
    "Explore the Platform",
    "Select a module below to begin exploring STX-LBD.",
)


# ---------- Row 1 ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("🔍 Search Hypotheses")
        st.write(
            "Search AI-ranked biological relationships and inspect the evidence "
            "supporting each predicted hypothesis."
        )
        st.markdown(
            """
            ✓ AI-ranked hypotheses  
            ✓ Biological interpretation  
            ✓ Temporal validation status  
            ✓ Supporting post-2015 literature  
            ✓ Bridge-node and model evidence
            """
        )

        if st.button(
            "Open Hypothesis Search →",
            key="open_search",
            use_container_width=True,
            type="primary",
        ):
            st.switch_page("pages/1_Search.py")


with col2:
    with st.container(border=True):
        st.subheader("🕸️ Knowledge Graph")
        st.write(
            "Explore semantic relationships among genes, toxins, taxa, "
            "environmental factors, biological processes, and detection methods."
        )
        st.markdown(
            """
            ✓ Interactive semantic network  
            ✓ Node-centered exploration  
            ✓ Entity-type filtering  
            ✓ Connected hypotheses  
            ✓ Graph exports
            """
        )

        if st.button(
            "Open Knowledge Graph →",
            key="open_graph",
            use_container_width=True,
        ):
            st.switch_page("pages/2_Knowledge_Graph.py")


with col3:
    with st.container(border=True):
        st.subheader("🧬 Cross-Taxa Explorer")
        st.write(
            "Compare literature-derived STX biology between dinoflagellates "
            "and cyanobacteria."
        )
        st.markdown(
            """
            ✓ Shared biological relationships  
            ✓ Lineage-specific associations  
            ✓ Transfer/convergence signals  
            ✓ Entity-level comparison  
            ✓ Shared biology network
            """
        )

        if st.button(
            "Open Cross-Taxa Explorer →",
            key="open_cross_taxa",
            use_container_width=True,
        ):
            st.switch_page("pages/3_Cross_Taxa.py")


# ---------- Row 2 ------------------------------------------------------------

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.subheader("📊 Discovery Analytics")
        st.write(
            "Explore interactive summaries of the corpus, knowledge graphs, "
            "hypothesis landscape, model performance, and temporal validation."
        )
        st.markdown(
            """
            ✓ Corpus and KG analytics  
            ✓ Hypothesis-class distribution  
            ✓ Validated vs unvalidated predictions  
            ✓ AI vs Node2Vec performance  
            ✓ Cross-taxa analytics
            """
        )

        if st.button(
            "Open Discovery Analytics →",
            key="open_analytics",
            use_container_width=True,
        ):
            st.switch_page("pages/4_Discovery_Analytics.py")


with col5:
    with st.container(border=True):
        st.subheader("📖 Documentation")
        st.write(
            "Review the STX-LBD methodology, interpretation guidance, "
            "data availability, software resources, and citation information."
        )
        st.markdown(
            """
            ✓ Framework overview  
            ✓ Methods and workflow  
            ✓ Interpretation guidance  
            ✓ GitHub and Zenodo resources  
            ✓ Citation information
            """
        )

        if st.button(
            "Open Documentation →",
            key="open_documentation",
            use_container_width=True,
        ):
            st.switch_page("pages/5_Documentation.py")


# ============================================================
# ABOUT STX-LBD
# ============================================================

section(
    "About STX-LBD",
    "From fragmented literature to explainable biological hypotheses.",
)

with st.container(border=True):
    st.markdown(
        """
        **STX-LBD** is an artificial intelligence-guided literature-based discovery
        framework developed for marine saxitoxin research. It transforms scientific
        literature into structured biological knowledge using semantic entity mining,
        knowledge graphs, graph representation learning, supervised machine learning,
        strict temporal validation, and cross-taxa knowledge transfer.

        The Explorer provides an interactive interface to investigate the resulting
        hypotheses and semantic relationships. Predictions are intended to support
        hypothesis generation and research prioritization rather than replace
        experimental validation.
        """
    )


# ============================================================
# ABOUT THE DEVELOPER
# ============================================================

section(
    "About the Developer",
    "Research and development behind the STX-LBD platform.",
)

with st.container(border=True):
    about_col1, about_col2 = st.columns([1, 2])

    with about_col1:
        st.markdown(
            """
            ### Buhari Lawan Muhammad

            **Marine biologist · Molecular ecologist · AI-assisted discovery researcher**
            """
        )

    with about_col2:
        st.markdown(
            """
            Buhari Lawan Muhammad develops research at the intersection of marine
            molecular ecology, harmful algal bloom biology, marine toxin research,
            evolutionary genomics, and artificial intelligence. His work uses
            molecular, transcriptomic, bioinformatic, and AI-guided approaches to
            investigate saxitoxin biosynthesis, toxin-producing dinoflagellates,
            environmental drivers, and biological knowledge discovery.

            STX-LBD was developed as an open research framework for converting the
            rapidly expanding marine saxitoxin literature into structured,
            testable biological hypotheses.
            """
        )


# ============================================================
# CITATION / RESOURCES
# ============================================================

section(
    "Citation & Resources",
    "Please cite the STX-LBD work when using the framework, data, or Explorer.",
)

with st.container(border=True):
    st.markdown(
        """
        **Suggested citation**

        **Buhari L. M. et al.** *STX-LBD: An AI-Guided Literature-Based Discovery
        Framework for Predicting Future Discoveries in Marine Saxitoxin Research.*

        *Manuscript currently under peer review. Update the citation after publication.*
        """
    )

    resource_col1, resource_col2 = st.columns(2)

    with resource_col1:
        st.link_button(
            "GitHub Repository",
            "https://github.com/buharilawan20-prog/STX-LBD",
            use_container_width=True,
        )

    with resource_col2:
        st.link_button(
            "Zenodo Archive",
            "https://doi.org/10.5281/zenodo.21640517",
            use_container_width=True,
        )


# ============================================================
# RESPONSIBLE USE
# ============================================================

st.info(
    "STX-LBD is a hypothesis-generation and prioritization resource. "
    "A temporally validated relationship indicates subsequent representation "
    "in post-2015 literature and should not automatically be interpreted as "
    "experimental proof of causality."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>STX-LBD Explorer</strong><br>
        Developed by Buhari Lawan Muhammad · Institute of Natural Science,
        Sangmyung University<br>
        AI-guided literature-based discovery for marine saxitoxin research
    </div>
    """,
    unsafe_allow_html=True,
)
