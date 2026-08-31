"""Root entry point for STX-LBD Explorer."""

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "scripts" / "proof_of_concept"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="STX-LBD Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------

HOME = st.Page(
    APP_DIR / "app.py",
    title="Home",
    icon="🏠",
    default=True,
)

SEARCH = st.Page(
    APP_DIR / "pages" / "1_Search.py",
    title="Search Hypotheses",
    icon="🔎",
)

KNOWLEDGE_GRAPH = st.Page(
    APP_DIR / "pages" / "2_Knowledge_Graph.py",
    title="Knowledge Graph",
    icon="🕸️",
)

CROSS_TAXA = st.Page(
    APP_DIR / "pages" / "3_Cross_Taxa.py",
    title="Cross-Taxa Explorer",
    icon="🧬",
)

ANALYTICS = st.Page(
    APP_DIR / "pages" / "4_Discovery_Analytics.py",
    title="Discovery Analytics",
    icon="📊",
)

DOCUMENTATION = st.Page(
    APP_DIR / "pages" / "5_Documentation.py",
    title="Documentation",
    icon="📘",
)


# ---------------------------------------------------------------------
# Expose page objects to homepage buttons
# ---------------------------------------------------------------------

st.session_state["_stxlbd_pages"] = {
    "home": HOME,
    "search": SEARCH,
    "knowledge_graph": KNOWLEDGE_GRAPH,
    "cross_taxa": CROSS_TAXA,
    "analytics": ANALYTICS,
    "documentation": DOCUMENTATION,
}


# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------

navigation = st.navigation(
    {
        "STX-LBD Explorer": [
            HOME,
            SEARCH,
            KNOWLEDGE_GRAPH,
            CROSS_TAXA,
            ANALYTICS,
            DOCUMENTATION,
        ]
    },
    position="sidebar",
)

navigation.run()
