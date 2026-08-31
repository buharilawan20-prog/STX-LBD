"""STX-LBD Explorer home page."""

import streamlit as st

from ui import (
    apply_global_style,
    footer,
    hero,
    module_card,
    render_sidebar,
    section,
)


# ---------------------------------------------------------------------
# Shared UI
# ---------------------------------------------------------------------

apply_global_style()
render_sidebar()


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------

hero(
    "STX-LBD Explorer",
    (
        "An interactive AI-assisted literature-based discovery environment "
        "for exploring hypotheses, knowledge graphs, cross-taxa relationships, "
        "and emerging discovery patterns in saxitoxin research."
    ),
    home=True,
)


# ---------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------

section(
    "Explore the Platform",
    "Select a module below to begin exploring STX-LBD.",
)


# ================================================================
# Row 1
# ================================================================

left, right = st.columns(2, gap="medium")


with left:

    module_card(
        "🔎",
        "Search Hypotheses",
        (
            "Search AI-ranked biological relationships and inspect the "
            "evidence supporting each predicted hypothesis."
        ),
        [
            "AI-ranked hypotheses",
            "Biological interpretation",
            "Temporal validation status",
            "Supporting post-2015 literature",
            "Bridge-node and model evidence",
        ],
    )

    if st.button(
        "Open Hypothesis Search →",
        key="home_search",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page(
            st.session_state["_stxlbd_pages"]["search"]
        )


with right:

    module_card(
        "🕸️",
        "Knowledge Graph",
        (
            "Explore semantic relationships among genes, toxins, taxa, "
            "environmental factors, biological processes, and detection methods."
        ),
        [
            "Interactive semantic network",
            "Node-centered exploration",
            "Entity-type filtering",
            "Connected hypotheses",
            "Graph exports",
        ],
    )

    if st.button(
        "Open Knowledge Graph →",
        key="home_kg",
        use_container_width=True,
    ):
        st.switch_page(
            st.session_state["_stxlbd_pages"]["knowledge_graph"]
        )


# ================================================================
# Row 2
# ================================================================

left, right = st.columns(2, gap="medium")


with left:

    module_card(
        "🧬",
        "Cross-Taxa Explorer",
        (
            "Compare dinoflagellate and cyanobacterial semantic knowledge "
            "graphs to investigate shared and lineage-specific relationships."
        ),
        [
            "Shared biological relationships",
            "Lineage-specific associations",
            "Cross-taxa transfer/convergence signals",
            "Entity-level comparison",
            "Shared biology network",
        ],
    )

    if st.button(
        "Open Cross-Taxa Explorer →",
        key="home_cross_taxa",
        use_container_width=True,
    ):
        st.switch_page(
            st.session_state["_stxlbd_pages"]["cross_taxa"]
        )


with right:

    module_card(
        "📊",
        "Discovery Analytics",
        (
            "Examine corpus statistics, hypothesis classes, temporal validation, "
            "AI performance, and discovery patterns."
        ),
        [
            "Corpus and graph statistics",
            "Validated vs unvalidated hypotheses",
            "AI model performance",
            "Temporal discovery patterns",
            "Cross-taxa analytics",
        ],
    )

    if st.button(
        "Open Discovery Analytics →",
        key="home_analytics",
        use_container_width=True,
    ):
        st.switch_page(
            st.session_state["_stxlbd_pages"]["analytics"]
        )


# ================================================================
# Row 3 — Documentation
# ================================================================

left, right = st.columns(2, gap="medium")


with left:

    module_card(
        "📘",
        "Documentation",
        (
            "Learn how STX-LBD was constructed, how predictions should be "
            "interpreted, and how to use the Explorer responsibly."
        ),
        [
            "Platform overview",
            "Methodological guidance",
            "Evidence definitions",
            "Interpretation guidance",
            "Responsible-use notes",
        ],
    )

    if st.button(
        "Open Documentation →",
        key="home_documentation",
        use_container_width=True,
    ):
        st.switch_page(
            st.session_state["_stxlbd_pages"]["documentation"]
        )


with right:

    st.markdown(
        """
<div class="info-card">
<h3>From Literature to Discovery</h3>
<p>
STX-LBD transforms fragmented literature into structured biological
knowledge and uses graph-based and supervised AI approaches to prioritize
potentially informative biological relationships.
</p>
<p>
The Explorer makes these predictions accessible through interactive
hypothesis search, semantic graph exploration, cross-taxa comparison,
supporting literature, and discovery analytics.
</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# About STX-LBD
# ---------------------------------------------------------------------

section(
    "About STX-LBD",
    (
        "A literature-based discovery framework for transforming fragmented "
        "saxitoxin research into structured, testable biological hypotheses."
    ),
)

st.write(
    """
STX-LBD integrates semantic knowledge graphs, graph representation learning,
supervised machine learning, strict temporal validation, and cross-taxa
knowledge transfer. The framework is designed to support scientific
hypothesis generation rather than replace experimental or field validation.
"""
)


# ---------------------------------------------------------------------
# About Developer
# ---------------------------------------------------------------------

section("About the Developer")

st.markdown(
    """
**Buhari Lawan Muhammad**

Research interests span marine molecular ecology, harmful algal bloom biology,
marine toxin research, evolutionary genomics, bioinformatics and
transcriptomics, and AI-assisted scientific discovery.

**Institute of Natural Science, Sangmyung University**  
Seoul, Republic of Korea
"""
)


# ---------------------------------------------------------------------
# Citation and resources
# ---------------------------------------------------------------------

section(
    "Citation & Resources",
    "Project resources and reproducibility information.",
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
**STX-LBD**

Buhari L. M. et al.  
*STX-LBD: An AI-Guided Literature-Based Discovery Framework for Marine
Saxitoxin Research.*

**GitHub**

https://github.com/buharilawan20-prog/STX-LBD
"""
    )

with col2:
    st.markdown(
        """
**Zenodo**

DOI: `10.5281/zenodo.21640517`

**Responsible use**

AI-ranked relationships represent hypotheses and literature-derived signals.
They should not be interpreted as causal biological evidence without
independent experimental, field, or computational validation.
"""
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

footer()
