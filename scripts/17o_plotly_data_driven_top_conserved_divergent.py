import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CROSS = BASE / "FINAL_WORKSPACE/cross_taxa"

OUT = BASE / "FINAL_WORKSPACE/figures"
OUT.mkdir(parents=True, exist_ok=True)

CONSERVED_FILE = CROSS / "cyano_all_vs_dino_post2015_convergent_edges.csv"

DIVERGENT_FILE = CROSS / "top_cyano_only_transfer_candidates.csv"

OUT_HTML = OUT / "Figure_top_data_driven_conserved_divergent_STX_plotly.html"

OUT_PDF = OUT / "Figure_top_data_driven_conserved_divergent_STX_plotly.pdf"

OUT_PNG = OUT / "Figure_top_data_driven_conserved_divergent_STX_plotly.png"

TABLE_OUT = OUT / "Figure_top_data_driven_conserved_divergent_STX_relationships_table.csv"

# ==========================================================
# SETTINGS
# ==========================================================

TOP_N_CONSERVED = 8
TOP_N_DIVERGENT = 8

# Generic semantic hubs to remove
HUB_TERMS = {
    "cyanobacteria",
    "cyanobacterial",
    "cyanobacterium",
    "dinoflagellate",
    "dinoflagellates",
    "toxin",
    "toxins",
    "toxic",
    "stx",
    "pst",
    "psts",
    "paralytic_shellfish_toxins",
    "paralytic_shellfish_poisoning"
}

# ==========================================================
# BIOLOGICAL CATEGORY KEYWORDS
# ==========================================================

ENV_TERMS = {
    "temperature",
    "salinity",
    "light",
    "nutrient",
    "nutrients",
    "nitrogen",
    "nitrate",
    "phosphorus",
    "phosphate",
    "bloom",
    "warming",
    "climate",
    "environment"
}

GENE_TERMS = {
    "sxta",
    "sxtg",
    "sxtd",
    "sxti",
    "sxtu",
    "sxth",
    "sxts",
    "gene",
    "genes",
    "expression",
    "transcriptome",
    "transcription"
}

EVOLUTION_TERMS = {
    "evolution",
    "phylogeny",
    "phylogenetic",
    "divergence",
    "conserved",
    "transfer",
    "adaptation"
}

MECHANISM_TERMS = {
    "biosynthesis",
    "toxin_biosynthesis",
    "saxitoxin_biosynthesis",
    "toxin_production",
    "regulation",
    "metabolism",
    "metabolic",
    "pathway",
    "mechanism",
    "functional",
    "arginine"
}

# ==========================================================
# HELPERS
# ==========================================================

def norm(x):
    return str(x).lower().strip().replace(" ", "_")

def clean(x):

    x = norm(x)

    repl = {
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "saxitoxin": "STX",
        "neosaxitoxin": "neoSTX",
        "gonyautoxin": "GTX",
        "paralytic_shellfish_toxins": "PSTs",
        "paralytic_shellfish_poisoning": "PSP",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "mass_spectrometry": "mass spectrometry",
        "mouse_bioassay": "mouse bioassay",
        "lc_ms": "LC-MS",
        "hplc": "HPLC"
    }

    return repl.get(x, x.replace("_", " "))

def assign_category(s, t):

    combined = f"{norm(s)} {norm(t)}"

    if any(k in combined for k in GENE_TERMS):
        return "Gene-related"

    if any(k in combined for k in ENV_TERMS):
        return "Environmental"

    if any(k in combined for k in EVOLUTION_TERMS):
        return "Evolutionary"

    if any(k in combined for k in MECHANISM_TERMS):
        return "Mechanistic"

    return "Other"

def has_hub(s, t):

    s = norm(s)
    t = norm(t)

    return s in HUB_TERMS or t in HUB_TERMS

# ==========================================================
# LOAD CONSERVED RELATIONSHIPS
# ==========================================================

def load_conserved():

    df = pd.read_csv(CONSERVED_FILE).fillna("")

    s_col = "source_cyano" if "source_cyano" in df.columns else "source"
    t_col = "target_cyano" if "target_cyano" in df.columns else "target"

    if "conservation_score" in df.columns:
        w_col = "conservation_score"

    elif "cyano_weight" in df.columns:
        w_col = "cyano_weight"

    elif "weight" in df.columns:
        w_col = "weight"

    else:
        w_col = None

    rows = []

    for _, r in df.iterrows():

        s = norm(r[s_col])
        t = norm(r[t_col])

        if not s or not t or s == t:
            continue

        if has_hub(s, t):
            continue

        category = assign_category(s, t)

        if category == "Other":
            continue

        weight = float(r[w_col]) if w_col else 1.0

        rows.append({
            "Category": category,
            "Relationship": f"{clean(s)} ↔ {clean(t)}",
            "Weight": weight,
            "Outcome": "Conserved"
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "No conserved relationships after filtering."
        )

    out = (
        out
        .groupby(
            ["Category", "Relationship", "Outcome"],
            as_index=False
        )
        .agg(Weight=("Weight", "sum"))
        .sort_values("Weight", ascending=False)
        .head(TOP_N_CONSERVED)
    )

    return out

# ==========================================================
# LOAD DIVERGENT RELATIONSHIPS
# ==========================================================

def load_divergent():

    df = pd.read_csv(DIVERGENT_FILE).fillna("")

    if "cyano_prior_weight" in df.columns:
        w_col = "cyano_prior_weight"

    elif "weight" in df.columns:
        w_col = "weight"

    else:
        w_col = None

    rows = []

    for _, r in df.iterrows():

        s = norm(r["source"])
        t = norm(r["target"])

        if not s or not t or s == t:
            continue

        if has_hub(s, t):
            continue

        category = assign_category(s, t)

        if category == "Other":
            continue

        weight = float(r[w_col]) if w_col else 1.0

        rows.append({
            "Category": category,
            "Relationship": f"{clean(s)} ↔ {clean(t)}",
            "Weight": weight,
            "Outcome": "Divergent"
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "No divergent relationships after filtering."
        )

    out = (
        out
        .groupby(
            ["Category", "Relationship", "Outcome"],
            as_index=False
        )
        .agg(Weight=("Weight", "sum"))
        .sort_values("Weight", ascending=False)
        .head(TOP_N_DIVERGENT)
    )

    return out

# ==========================================================
# LOAD DATA
# ==========================================================

conserved = load_conserved()
divergent = load_divergent()

plot_df = pd.concat(
    [conserved, divergent],
    ignore_index=True
)

# Save table used
plot_df.to_csv(
    TABLE_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# BUILD SANKEY
# ==========================================================

labels = []

def add_node(label):

    if label not in labels:
        labels.append(label)

    return labels.index(label)

sources = []
targets = []
values = []
link_colors = []

category_colors = {
    "Environmental": "rgba(230,159,0,0.85)",
    "Evolutionary": "rgba(142,68,173,0.85)",
    "Gene-related": "rgba(0,158,115,0.85)",
    "Mechanistic": "rgba(204,121,167,0.85)",
    "Other": "rgba(150,150,150,0.70)"
}

outcome_colors = {
    "Conserved": "rgba(42,157,143,0.90)",
    "Divergent": "rgba(231,111,81,0.90)"
}

for _, r in plot_df.iterrows():

    cat = r["Category"]

    rel = (
        f'{r["Relationship"]}'
        f'<br>W={r["Weight"]:.0f}'
    )

    outcome = r["Outcome"]

    val = float(r["Weight"])

    i_cat = add_node(cat)
    i_rel = add_node(rel)
    i_out = add_node(outcome)

    # category -> relationship
    sources.append(i_cat)
    targets.append(i_rel)
    values.append(val)
    link_colors.append(
        category_colors.get(
            cat,
            "rgba(150,150,150,0.50)"
        )
    )

    # relationship -> outcome
    sources.append(i_rel)
    targets.append(i_out)
    values.append(val)
    link_colors.append(
        outcome_colors[outcome]
    )

# ==========================================================
# NODE COLORS
# ==========================================================

node_colors = []

for lab in labels:

    clean_lab = lab.split("<br>")[0]

    if clean_lab in category_colors:
        node_colors.append(
            category_colors[clean_lab]
        )

    elif clean_lab in outcome_colors:
        node_colors.append(
            outcome_colors[clean_lab]
        )

    else:
        node_colors.append(
            "rgba(245,245,245,1.0)"
        )

# ==========================================================
# PLOTLY FIGURE
# ==========================================================

fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",

            node=dict(
                pad=24,
                thickness=24,
                line=dict(
                    color="black",
                    width=0.5
                ),
                label=labels,
                color=node_colors
            ),

            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )
    ]
)

fig.update_layout(
    title=dict(
        text=(
            "Top data-driven conserved and divergent "
            "STX relationships"
        ),
        x=0.5,
        font=dict(size=24)
    ),

    font=dict(size=13),

    width=1500,
    height=900
)

# ==========================================================
# SAVE
# ==========================================================

fig.write_html(str(OUT_HTML))

# Requires:
# pip install kaleido
fig.write_image(
    str(OUT_PDF),
    format="pdf",
    scale=3
)

fig.write_image(
    str(OUT_PNG),
    format="png",
    scale=4
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\nSaved:")
print(OUT_HTML)
print(OUT_PDF)
print(OUT_PNG)
print(TABLE_OUT)

print("\nRelationships used:")
print(plot_df.to_string(index=False))
