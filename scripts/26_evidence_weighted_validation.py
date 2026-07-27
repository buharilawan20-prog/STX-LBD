import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CANDIDATES = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_candidates.csv"
POST_EDGES = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT.mkdir(parents=True, exist_ok=True)

OUT_VALIDATED = OUT / "top_validated_hypotheses_evidence_weighted.csv"
OUT_UNVALIDATED = OUT / "top_unvalidated_hypotheses_refined.csv"

TOP_N = 50

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "degree_u",
    "degree_v"
]

# ==========================================================
# HELPERS
# ==========================================================

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    return "||".join(sorted([norm(a), norm(b)]))

def clean_label(x):
    x = norm(x)

    repl = {
        "sxta": "sxtA",
        "sxta4": "sxtA4",
        "sxtb": "sxtB",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth": "sxtH",
        "sxts": "sxtS",
        "sxt_genes": "sxt genes",
        "saxitoxin": "STX",
        "gtx": "GTX",
        "gonyautoxin": "GTX",
        "neosaxitoxin": "neoSTX",
        "paralytic_shellfish_toxins": "PSTs",
        "paralytic_shellfish_poisoning": "PSP",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "stx_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "gene_expression": "gene expression",
        "mass_spectrometry": "mass spectrometry",
        "mouse_bioassay": "mouse bioassay",
        "lc_ms": "LC-MS",
        "hplc": "HPLC",
        "elisa": "ELISA",
        "alexandrium_catenella": "Alexandrium catenella",
        "alexandrium_minutum": "Alexandrium minutum",
        "alexandrium_fundyense": "Alexandrium fundyense",
        "alexandrium_tamarense": "Alexandrium tamarense",
        "alexandrium_pacificum": "Alexandrium pacificum",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
        "pyrodinium_bahamense": "Pyrodinium bahamense"
    }

    return repl.get(x, x.replace("_", " "))

def infer_category(a, b):
    pair = f"{norm(a)} {norm(b)}"

    if any(x in pair for x in ["sxta", "sxtb", "sxtg", "sxtd", "sxti", "sxtu", "sxth", "sxts", "sxt_genes"]):
        return "Gene-related"

    if any(x in pair for x in ["nitrogen", "nitrate", "nutrient", "phosphorus", "phosphate", "light", "salinity", "temperature", "warming"]):
        return "Environmental regulation"

    if any(x in pair for x in ["biosynthesis", "toxin_production", "gene_expression", "expression", "regulation"]):
        return "Mechanistic"

    if any(x in pair for x in ["phylogenetic", "evolution", "gene_loss"]):
        return "Evolutionary"

    if any(x in pair for x in ["mass_spectrometry", "lc_ms", "hplc", "elisa", "mouse_bioassay"]):
        return "Detection / profiling"

    if any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]):
        return "Taxon-associated"

    return "Other"

def biological_interpretation(a, b):
    pair = f"{norm(a)} {norm(b)}"

    if any(x in pair for x in ["nitrate", "nitrogen", "nutrient"]) and any(x in pair for x in ["sxt", "biosynthesis", "toxin"]):
        return "Suggests nutrient-mediated regulation of sxt genes and saxitoxin biosynthesis."

    if any(x in pair for x in ["phosphate", "phosphorus"]) and any(x in pair for x in ["sxt", "biosynthesis", "toxin"]):
        return "Suggests phosphorus-linked modulation of toxin biosynthesis or toxin-related gene dynamics."

    if any(x in pair for x in ["salinity", "light", "temperature", "warming"]) and "sxt" in pair:
        return "Suggests environmental modulation of saxitoxin biosynthesis gene dynamics."

    if any(x in pair for x in ["salinity", "light", "temperature", "warming"]) and any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]):
        return "Suggests environmental association with toxic dinoflagellate ecology or distribution."

    if "bloom" in pair and any(x in pair for x in ["sxt", "gtx", "toxin", "biosynthesis"]):
        return "Suggests bloom-associated toxin gene or toxin phenotype dynamics."

    if any(x in pair for x in ["gtx", "gonyautoxin", "neosaxitoxin", "saxitoxin"]) and any(x in pair for x in ["biosynthesis", "expression", "production"]):
        return "Links toxin analog profiles with biosynthetic or regulatory processes."

    if any(x in pair for x in ["sxt"]) and any(x in pair for x in ["biosynthesis", "production", "regulation", "expression"]):
        return "Supports mechanistic coupling between sxt genes and toxin biosynthesis or regulation."

    if "alexandrium" in pair and "sxt" in pair:
        return "Indicates taxon-specific association between Alexandrium species and sxt gene repertoire."

    if any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]) and any(x in pair for x in ["gtx", "neosaxitoxin", "saxitoxin", "toxin"]):
        return "Represents species-level association with saxitoxin or toxin analog production."

    if any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]) and any(x in pair for x in ["mass_spectrometry", "lc_ms", "hplc", "elisa", "mouse_bioassay"]):
        return "Reflects toxin detection, analytical profiling, or validation workflows for toxic dinoflagellates."

    if any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]) and any(x in pair for x in ["alexandrium", "gymnodinium", "pyrodinium"]):
        return "Suggests comparative ecological, taxonomic, or evolutionary association among toxin-producing dinoflagellates."

    if any(x in pair for x in ["phylogenetic", "evolution", "gene_loss"]) and "sxt" in pair:
        return "Suggests evolutionary restructuring, retention, or loss of saxitoxin-related genes."

    if any(x in pair for x in ["phylogenetic", "evolution"]) and any(x in pair for x in ["temperature", "salinity", "warming"]):
        return "Suggests environmental gradients may be linked to evolutionary or phylogenetic structuring of toxic dinoflagellates."

    if any(x in pair for x in ["mass_spectrometry", "lc_ms", "hplc", "elisa"]) and any(x in pair for x in ["temperature", "nutrient", "nitrogen", "phosphorus"]):
        return "Indicates analytical investigation of environmentally driven toxin dynamics."

    return "Represents a lower-specificity semantic association within dinoflagellate saxitoxin biology."

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading strict candidates and post-2015 edges...")

df = pd.read_csv(CANDIDATES).fillna("")
post = pd.read_csv(POST_EDGES).fillna("")

# ==========================================================
# TRAIN LOGISTIC MODEL TO SCORE CANDIDATES
# ==========================================================

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

df["AI_Probability"] = model.predict_proba(X)[:, 1]

# ==========================================================
# BUILD POST-2015 EVIDENCE COUNTS
# ==========================================================

post["pair"] = post.apply(
    lambda r: pair_key(r["source"], r["target"]),
    axis=1
)

# Try to infer document/paper column
paper_cols = [
    "document_id", "paper_id", "pmid", "doi", "title", "source_file"
]

paper_col = None
for c in paper_cols:
    if c in post.columns:
        paper_col = c
        break

# Try to infer year column
year_col = None
for c in ["year", "Year", "publication_year"]:
    if c in post.columns:
        year_col = c
        break

evidence_rows = []

for pair, g in post.groupby("pair"):

    if paper_col:
        n_papers = g[paper_col].astype(str).nunique()
    else:
        n_papers = len(g)

    if year_col:
        yrs = pd.to_numeric(g[year_col], errors="coerce").dropna()
        earliest_year = int(yrs.min()) if len(yrs) else ""
    else:
        earliest_year = ""

    evidence_rows.append({
        "pair": pair,
        "Supporting_Post2015_Papers": n_papers,
        "Earliest_Validation_Year": earliest_year
    })

evidence = pd.DataFrame(evidence_rows)

# ==========================================================
# MERGE EVIDENCE WITH CANDIDATES
# ==========================================================

df["pair"] = df.apply(
    lambda r: pair_key(r["source"], r["target"]),
    axis=1
)

df = df.merge(
    evidence,
    on="pair",
    how="left"
)

df["Supporting_Post2015_Papers"] = df["Supporting_Post2015_Papers"].fillna(0).astype(int)
df["Earliest_Validation_Year"] = df["Earliest_Validation_Year"].fillna("")

# ==========================================================
# ADD LABELS AND INTERPRETATION
# ==========================================================

df["Source_Label"] = df["source"].apply(clean_label)
df["Target_Label"] = df["target"].apply(clean_label)
df["Hypothesis"] = df["Source_Label"] + " ↔ " + df["Target_Label"]

df["Biological_Category"] = df.apply(
    lambda r: infer_category(r["source"], r["target"]),
    axis=1
)

df["Biological_Interpretation"] = df.apply(
    lambda r: biological_interpretation(r["source"], r["target"]),
    axis=1
)

# ==========================================================
# OUTPUT VALIDATED
# ==========================================================

validated = (
    df[df["label"] == 1]
    .sort_values(
        ["AI_Probability", "Supporting_Post2015_Papers"],
        ascending=False
    )
    .head(TOP_N)
    .copy()
)

validated.insert(0, "Rank", range(1, len(validated) + 1))
validated["Validation_Status"] = "Validated in post-2015 literature"

# ==========================================================
# OUTPUT UNVALIDATED
# ==========================================================

unvalidated = (
    df[df["label"] == 0]
    .sort_values("AI_Probability", ascending=False)
    .head(TOP_N)
    .copy()
)

unvalidated.insert(0, "Rank", range(1, len(unvalidated) + 1))
unvalidated["Validation_Status"] = "Unvalidated / candidate future hypothesis"

# ==========================================================
# SAVE
# ==========================================================

cols = [
    "Rank",
    "Hypothesis",
    "Validation_Status",
    "AI_Probability",
    "Supporting_Post2015_Papers",
    "Earliest_Validation_Year",
    "Biological_Category",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "Biological_Interpretation"
]

validated[cols].to_csv(
    OUT_VALIDATED,
    index=False,
    encoding="utf-8-sig"
)

unvalidated[cols].to_csv(
    OUT_UNVALIDATED,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUT_VALIDATED)
print(OUT_UNVALIDATED)

print("\nEvidence counted using paper column:", paper_col if paper_col else "None; edge count used")
print("Year column:", year_col if year_col else "None")

print("\nTop validated hypotheses:")
print(validated[cols].head(20).to_string(index=False))

print("\nTop unvalidated hypotheses:")
print(unvalidated[cols].head(20).to_string(index=False))
