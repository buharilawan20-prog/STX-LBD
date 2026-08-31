"""Shared UI components for STX-LBD Explorer."""

import html

import streamlit as st


# ---------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------

GLOBAL_CSS = """
<style>

/* =========================
   Global
   ========================= */

:root {
    --navy: #0B1F3A;
    --navy2: #163D77;
    --blue: #1F5FAF;
    --blue-light: #EAF2FF;
    --surface: #FFFFFF;
    --background: #F6F8FC;
    --text: #182230;
    --muted: #667085;
    --border: #DCE4EF;
    --shadow: rgba(15, 35, 65, 0.07);
}

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: var(--background);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}


/* =========================
   Sidebar
   ========================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0B1F3A 0%,
        #153B70 100%
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.16);
}

[data-testid="stSidebarNav"] {
    padding-top: 0.25rem;
}

[data-testid="stSidebarNav"] a {
    border-radius: 9px;
    margin-bottom: 0.18rem;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.09);
}

.stx-brand {
    padding: 0.15rem 0 0.75rem 0;
}

.stx-brand-title {
    font-size: 1.35rem;
    font-weight: 760;
    letter-spacing: -0.02em;
    margin-bottom: 0.45rem;
}

.stx-brand-subtitle {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.72);
    line-height: 1.55;
}


/* =========================
   Hero
   ========================= */

.page-hero {
    background: linear-gradient(
        135deg,
        #0B1F3A 0%,
        #163D77 100%
    );
    border-radius: 20px;
    padding: 2.5rem 2.25rem;
    margin-bottom: 1.55rem;
    box-shadow: 0 12px 30px rgba(11,31,58,0.13);
}

.page-hero h1 {
    color: white;
    margin: 0;
    font-size: 2.45rem;
    line-height: 1.15;
    font-weight: 770;
    letter-spacing: -0.035em;
}

.page-hero p {
    color: #E6EDF9;
    max-width: 900px;
    margin: 0.8rem 0 0;
    line-height: 1.68;
    font-size: 1rem;
}

.home-hero {
    text-align: center;
}

.home-hero p {
    margin-left: auto;
    margin-right: auto;
}


/* =========================
   Section headings
   ========================= */

.section-heading {
    margin: 2.1rem 0 1rem;
}

.section-heading h2 {
    margin: 0;
    color: var(--navy);
    font-size: 1.62rem;
    font-weight: 735;
    letter-spacing: -0.02em;
}

.section-heading p {
    margin: 0.35rem 0 0;
    color: var(--muted);
    line-height: 1.55;
}


/* =========================
   Module cards
   ========================= */

.tool-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.45rem 1.4rem;
    min-height: 290px;
    box-shadow: 0 5px 16px var(--shadow);
    margin-bottom: 0.15rem;
}

.tool-card:hover {
    border-color: #B9CAE1;
    box-shadow: 0 8px 23px rgba(15,35,65,0.10);
}

.tool-card h3 {
    margin: 0 0 0.75rem;
    color: var(--navy);
    font-size: 1.32rem;
    font-weight: 725;
}

.tool-card p {
    color: #40516A;
    line-height: 1.62;
    margin: 0 0 0.9rem;
}

.tool-features {
    color: #40516A;
    line-height: 1.75;
    font-size: 0.91rem;
}


/* =========================
   Buttons
   ========================= */

div.stButton > button {
    min-height: 44px;
    border-radius: 10px;
    font-weight: 650;
}


/* =========================
   Information cards
   ========================= */

.info-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.25rem 1.35rem;
    box-shadow: 0 4px 14px var(--shadow);
    height: 100%;
}

.info-card h3 {
    color: var(--navy);
    margin-top: 0;
    margin-bottom: 0.65rem;
}

.info-card p {
    color: #40516A;
    line-height: 1.65;
}


/* =========================
   Metrics / tables
   ========================= */

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


/* =========================
   Footer
   ========================= */

.stx-footer {
    margin-top: 3rem;
    padding: 1.6rem 1rem 0.4rem;
    border-top: 1px solid var(--border);
    text-align: center;
}

.stx-footer-name {
    color: var(--navy);
    font-weight: 700;
    font-size: 0.95rem;
}

.stx-footer-affiliation {
    color: var(--muted);
    margin-top: 0.25rem;
    font-size: 0.86rem;
}

.stx-footer-project {
    color: #8A96A8;
    margin-top: 0.3rem;
    font-size: 0.8rem;
}

</style>
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def apply_global_style() -> None:
    """Apply shared STX-LBD Explorer CSS."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render STX-LBD branding above the native navigation."""

    with st.sidebar:
        st.markdown(
            """
<div class="stx-brand">
<div class="stx-brand-title">🧬 STX-LBD</div>
<div class="stx-brand-subtitle">
AI-guided literature-based discovery for marine saxitoxin research.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.caption(
            "Interactive AI-assisted scientific discovery"
        )


def hero(
    title: str,
    description: str,
    home: bool = False,
) -> None:
    """Render page hero."""

    css_class = "page-hero home-hero" if home else "page-hero"

    st.markdown(
        f"""
<div class="{css_class}">
<h1>{html.escape(title)}</h1>
<p>{html.escape(description)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def section(
    title: str,
    description: str = "",
) -> None:
    """Render consistent section heading."""

    title_safe = html.escape(title)

    if description:
        description_safe = html.escape(description)
        body = (
            f'<div class="section-heading">'
            f'<h2>{title_safe}</h2>'
            f'<p>{description_safe}</p>'
            f'</div>'
        )
    else:
        body = (
            f'<div class="section-heading">'
            f'<h2>{title_safe}</h2>'
            f'</div>'
        )

    st.markdown(body, unsafe_allow_html=True)


def module_card(
    icon: str,
    title: str,
    description: str,
    features: list[str],
) -> None:
    """Render one platform module card."""

    feature_html = "<br>".join(
        f"✓ {html.escape(feature)}"
        for feature in features
    )

    card = (
        '<div class="tool-card">'
        f'<h3>{icon} {html.escape(title)}</h3>'
        f'<p>{html.escape(description)}</p>'
        f'<div class="tool-features">{feature_html}</div>'
        '</div>'
    )

    st.markdown(card, unsafe_allow_html=True)


def footer() -> None:
    """Render project/developer footer."""

    st.markdown(
        """
<div class="stx-footer">
<div class="stx-footer-name">
Buhari Lawan Muhammad
</div>
<div class="stx-footer-affiliation">
Institute of Natural Science · Sangmyung University · Seoul, Republic of Korea
</div>
<div class="stx-footer-project">
STX-LBD Explorer · AI-assisted literature-based discovery for marine saxitoxin research
</div>
</div>
""",
        unsafe_allow_html=True,
    )
