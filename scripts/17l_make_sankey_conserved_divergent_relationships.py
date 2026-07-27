import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CROSS_DIR = BASE / "FINAL_WORKSPACE/cross_taxa"
OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONSERVED_FILE = CROSS_DIR / "cyano_all_vs_dino_post2015_convergent_edges.csv"
DIVERGENT_FILE = CROSS_DIR / "top_cyano_only_transfer_candidates.csv"

OUT_HTML = OUT_DIR / "Figure_sankey_conserved_divergent_STX_relationships.html"
OUT_PNG = OUT_DIR / "Figure_sankey_conserved_divergent_STX_relationships.png"

TOP_N = 12

def normalize(x):
    return str(x).lower().strip().replace(" ", "_")

def clean_label(x):
    x = str(x)
    repl = {
        "paralytic_shellfish_toxins": "PSTs",
        "paralytic_shellfish_poisoning": "PSP",
        "saxitoxin": "saxitoxin",
        "toxin_production": "toxin production",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "mass_spectrometry": "mass spectrometry",
        "cyanobacteria": "cyanobacteria",
        "dinoflagellate": "dinoflagellate",
        "alexandrium": "Alexandrium",
        "gymnodinium": "Gymnodinium",
        "pyrodinium": "Pyrodinium",
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
    }
    return repl.get(x, x.replace("_", " "))

ENV_TERMS = {
    "temperature", "salinity", "light", "nutrient", "nutrients",
    "nitrogen", "nitrate", "phosphorus", "phosphate", "bloom",
    "warming", "climate", "environment"
}

GENE_TERMS = {
    "sxta", "sxtg", "sxtd", "sxti", "sxt", "gene", "genes",
    "expression", "transcriptome", "transcription"
}

EVOLUTION_TERMS = {
    "evolution", "phylogeny", "phylogenetic", "divergence",
    "conserved", "transfer", "adaptation"
}

MECHANISTIC_TERMS = {
    "biosynthesis", "toxin_biosynthesis", "saxitoxin_biosynthesis",
    "toxin_production", "regulation", "metabolism", "metabolic",
    "pathway", "mechanism", "functional"
}

def assign_category(s, t):
    combined = f"{normalize(s)} {normalize(t)}"

    if any(term in combined for term in ENV_TERMS):
        return "Environmental"
    if any(term in combined for term in GENE_TERMS):
        return "Gene-related"
    if any(term in combined for term in EVOLUTION_TERMS):
        return "Evolutionary"
    if any(term in combined for term in MECHANISTIC_TERMS):
        return "Mechanistic"

    return "Other"

def prepare_conserved():
    df = pd.read_csv(CONSERVED_FILE).fillna("")

    source_col = "source_cyano" if "source_cyano" in df.columns else "source"
    target_col = "target_cyano" if "target_cyano" in df.columns else "target"

    weight_col = "conservation_score" if "conservation_score" in df.columns else "cyano_weight"

    if weight_col not in df.columns:
        weight_col = "weight" if "weight" in df.columns else None

    rows = []

    for _, r in df.iterrows():
        s = normalize(r[source_col])
        t = normalize(r[target_col])

        if not s or not t:
            continue

        weight = float(r[weight_col]) if weight_col else 1.0
        category = assign_category(s, t)

        if category == "Other":
            continue

        rel = f"{clean_label(s)} ↔ {clean_label(t)}"

        rows.append({
            "Category": category,
            "Class": "Conserved / transferred",
            "Relationship": rel,
            "Weight": weight
        })

    out = pd.DataFrame(rows)

    return (
        out.groupby(["Category", "Class", "Relationship"], as_index=False)
        .agg(Weight=("Weight", "sum"))
        .sort_values("Weight", ascending=False)
        .head(TOP_N)
    )

def prepare_divergent():
    df = pd.read_csv(DIVERGENT_FILE).fillna("")

    weight_col = "cyano_prior_weight" if "cyano_prior_weight" in df.columns else None

    rows = []

    for _, r in df.iterrows():
        s = normalize(r["source"])
        t = normalize(r["target"])

        if not s or not t:
            continue

        weight = float(r[weight_col]) if weight_col else 1.0
        category = assign_category(s, t)

        if category == "Other":
            continue

        rel = f"{clean_label(s)} ↔ {clean_label(t)}"

        rows.append({
            "Category": category,
            "Class": "Divergent / cyano-only",
            "Relationship": rel,
            "Weight": weight
        })

    out = pd.DataFrame(rows)

    return (
        out.groupby(["Category", "Class", "Relationship"], as_index=False)
        .agg(Weight=("Weight", "sum"))
        .sort_values("Weight", ascending=False)
        .head(TOP_N)
    )

conserved = prepare_conserved()
divergent = prepare_divergent()

plot_df = pd.concat([conserved, divergent], ignore_index=True)

if plot_df.empty:
    raise ValueError("No relationships found for Sankey plot.")

# Build Sankey nodes
labels = []

def add_label(x):
    if x not in labels:
        labels.append(x)
    return labels.index(x)

sources = []
targets = []
values = []

for _, r in plot_df.iterrows():
    cat = r["Category"]
    cls = r["Class"]
    rel = r["Relationship"]
    val = float(r["Weight"])

    i_cat = add_label(cat)
    i_cls = add_label(cls)
    i_rel = add_label(rel)

    sources.append(i_cat)
    targets.append(i_cls)
    values.append(val)

    sources.append(i_cls)
    targets.append(i_rel)
    values.append(val)

fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=20,
                thickness=18,
                line=dict(color="black", width=0.4),
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )
    ]
)

fig.update_layout(
    title_text="Top conserved and divergent STX semantic relationships",
    font_size=12,
    width=1300,
    height=850
)

fig.write_html(str(OUT_HTML))

try:
    fig.write_image(str(OUT_PNG), scale=3)
    print("PNG saved:", OUT_PNG)
except Exception as e:
    print("PNG export skipped. Install kaleido if needed:")
    print("pip install -U kaleido")
    print("Error:", e)

print("\nSaved:")
print(OUT_HTML)

print("\nRelationships used:")
print(plot_df.to_string(index=False))
