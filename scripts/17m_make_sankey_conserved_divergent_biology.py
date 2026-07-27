import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "FINAL_WORKSPACE/cross_taxa/true_divergent_vs_conserved_category_counts.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HTML = OUT_DIR / "Figure_sankey_conserved_divergent_STX_biology.html"

df = pd.read_csv(INPUT)

order = ["Environmental", "Evolutionary", "Gene-related", "Mechanistic"]
df = df[df["Category"].isin(order)].copy()

labels = [
    "Cyanobacterial STX knowledge",
    "Conserved / transferred biology",
    "Divergent / cyano-only biology",
    "Environmental",
    "Evolutionary",
    "Gene-related",
    "Mechanistic"
]

idx = {lab: i for i, lab in enumerate(labels)}

sources = []
targets = []
values = []

# Cyano knowledge -> relationship classes
total_conserved = df["Conserved_Count"].sum()
total_divergent = df["Divergent_Count"].sum()

sources += [idx["Cyanobacterial STX knowledge"], idx["Cyanobacterial STX knowledge"]]
targets += [idx["Conserved / transferred biology"], idx["Divergent / cyano-only biology"]]
values += [total_conserved, total_divergent]

# Relationship class -> biological categories
for _, row in df.iterrows():

    cat = row["Category"]

    conserved = int(row["Conserved_Count"])
    divergent = int(row["Divergent_Count"])

    sources.append(idx["Conserved / transferred biology"])
    targets.append(idx[cat])
    values.append(conserved)

    sources.append(idx["Divergent / cyano-only biology"])
    targets.append(idx[cat])
    values.append(divergent)

node_colors = [
    "#56B4E9",
    "#FF7F0E",
    "#1F77B4",
    "#E69F00",
    "#8E44AD",
    "#009E73",
    "#CC79A7"
]

link_colors = []

for s, t in zip(sources, targets):
    if s == idx["Conserved / transferred biology"] or t == idx["Conserved / transferred biology"]:
        link_colors.append("rgba(255,127,14,0.45)")
    elif s == idx["Divergent / cyano-only biology"] or t == idx["Divergent / cyano-only biology"]:
        link_colors.append("rgba(31,119,180,0.45)")
    else:
        link_colors.append("rgba(120,120,120,0.35)")

fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=25,
                thickness=22,
                line=dict(color="black", width=0.5),
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
    title_text="Conserved and divergent STX biology between cyanobacteria and dinoflagellates",
    font_size=14,
    width=1200,
    height=750
)

fig.write_html(str(OUT_HTML))

print("\nSaved:")
print(OUT_HTML)

print("\nSummary used:")
print(df[["Category", "Conserved_Count", "Divergent_Count"]].to_string(index=False))
print("\nTotal conserved:", total_conserved)
print("Total divergent:", total_divergent)
