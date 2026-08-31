
import streamlit as st

GLOBAL_CSS = """
<style>
:root {
    --navy: #0B1F3A;
    --blue: #1F5FAF;
    --light-blue: #EAF2FF;
    --surface: #FFFFFF;
    --background: #F6F8FC;
    --text: #182230;
    --muted: #667085;
    --border: #DCE4EF;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: var(--background);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.35rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F3A 0%, #163D77 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

div.stButton > button {
    min-height: 44px;
    border-radius: 10px;
    font-weight: 650;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 0.8rem 1rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.page-hero {
    background: linear-gradient(135deg, #0B1F3A, #163D77);
    color: white;
    border-radius: 20px;
    padding: 2.4rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 28px rgba(11,31,58,0.14);
}

.page-hero h1 {
    margin: 0;
    font-size: 2.35rem;
    font-weight: 760;
}

.page-hero p {
    max-width: 860px;
    margin: 0.75rem 0 0;
    color: #E6EDF9;
    line-height: 1.65;
}

.home-hero {
    text-align: center;
}

.home-hero p {
    margin-left: auto;
    margin-right: auto;
}

.section-heading {
    margin: 2rem 0 1rem;
}

.section-heading h2 {
    margin: 0;
    color: var(--navy);
    font-size: 1.65rem;
}

.section-heading p {
    margin: 0.35rem 0 0;
    color: var(--muted);
}

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15,35,65,0.05);
}

.metric-number {
    color: var(--blue);
    font-size: 2rem;
    font-weight: 760;
}

.metric-label {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

.tool-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.45rem;
    min-height: 235px;
    box-shadow: 0 4px 14px rgba(15,35,65,0.05);
}

.tool-card h3 {
    margin: 0 0 0.65rem;
    color: var(--navy);
}

.tool-card p {
    color: var(--muted);
    line-height: 1.62;
}

.tool-features {
    margin-top: 0.8rem;
    color: #40516A;
    line-height: 1.75;
    font-size: 0.9rem;
}

.footer {
    margin-top: 2.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    text-align: center;
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.6;
}
</style>
"""

def apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def render_sidebar() -> None:
    with st.sidebar:
        st.title("🧬 STX-LBD")
        st.caption(
            "AI-guided literature-based discovery for marine saxitoxin research."
        )
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Search.py", label="Search Hypotheses", icon="🔍")
        st.page_link("pages/2_Knowledge_Graph.py", label="Knowledge Graph", icon="🕸️")
        st.page_link("pages/3_Cross_Taxa.py", label="Cross-Taxa Explorer", icon="🧬")
        st.page_link("pages/4_Discovery_Analytics.py", label="Discovery Analytics", icon="📊")
        st.page_link("pages/5_Documentation.py", label="Documentation", icon="📖")
        st.divider()
        st.caption(
            "Predicted relationships are research hypotheses and require independent validation."
        )

def hero(title: str, description: str, home: bool = False) -> None:
    extra = " home-hero" if home else ""
    st.markdown(
        f"""
        <div class="page-hero{extra}">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section(title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
