import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CANDIDATES = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_candidates.csv"
POST_EDGES = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT_VALIDATED = OUT / "actionable_top_validated_hypotheses.csv"
OUT_UNVALIDATED = OUT / "actionable_top_unvalidated_hypotheses.csv"

TOP_N = 30

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "degree_u",
    "degree_v"
]

GENES = {"sxta","sxta4","sxtb","sxtg","sxtd","sxti","sxtu","sxth","sxts","sxt_genes"}
ENV = {"nitrogen","nitrate","nutrient","nutrients","phosphorus","phosphate","light","salinity","temperature","warming"}
TOXINS = {"gtx","gonyautoxin","neosaxitoxin","saxitoxin","stx"}
PROCESSES = {"stx_biosynthesis","saxitoxin_biosynthesis","toxin_biosynthesis","toxin_production","gene_expression","expression","regulation","biosynthetic_pathway"}
TAXA = {"alexandrium","alexandrium_catenella","alexandrium_minutum","alexandrium_fundyense","alexandrium_tamarense","alexandrium_pacificum","gymnodinium_catenatum","pyrodinium_bahamense"}
BAD_TERMS = {"mass_spectrometry","lc_ms","hplc","elisa","mouse_bioassay","phylogenetic","pyrodinium","gymnodinium"}

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    return "||".join(sorted([norm(a), norm(b)]))

def clean_label(x):
    x = norm(x)
    repl = {
        "sxta": "sxtA", "sxta4": "sxtA4", "sxtb": "sxtB", "sxtg": "sxtG",
        "sxtd": "sxtD", "sxti": "sxtI", "sxtu": "sxtU", "sxth": "sxtH",
        "sxts": "sxtS", "sxt_genes": "sxt genes",
        "gtx": "GTX", "gonyautoxin": "GTX", "neosaxitoxin": "neoSTX",
        "saxitoxin": "STX", "stx_biosynthesis": "STX biosynthesis",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "gene_expression": "gene expression",
        "alexandrium_catenella": "Alexandrium catenella",
        "alexandrium_minutum": "Alexandrium minutum",
        "alexandrium_fundyense": "Alexandrium fundyense",
        "alexandrium_tamarense": "Alexandrium tamarense",
        "alexandrium_pacificum": "Alexandrium pacificum",
        "gymnodinium_catenatum": "Gymnodinium catenatum",
        "pyrodinium_bahamense": "Pyrodinium bahamense"
    }
    return repl.get(x, x.replace("_", " "))

def category(x):
    x = norm(x)
    if x in GENES or x.startswith("sxt"):
        return "GENE"
    if x in ENV:
        return "ENV"
    if x in TOXINS:
        return "TOXIN"
    if x in PROCESSES or "biosynthesis" in x or "expression" in x or "regulation" in x:
        return "PROCESS"
    if x in TAXA or x.startswith("alexandrium"):
        return "TAXON"
    if x in BAD_TERMS:
        return "BAD"
    return "OTHER"

def actionable(a, b):
    a, b = norm(a), norm(b)
    ca, cb = category(a), category(b)

    if a in BAD_TERMS or b in BAD_TERMS:
        return False

    good_pairs = {
        ("ENV", "GENE"),
        ("ENV", "TOXIN"),
        ("ENV", "PROCESS"),
        ("GENE", "TOXIN"),
        ("GENE", "PROCESS"),
        ("TAXON", "GENE"),
        ("TAXON", "TOXIN"),
        ("TAXON", "PROCESS"),
        ("PROCESS", "TOXIN"),
    }

    return (ca, cb) in good_pairs or (cb, ca) in good_pairs

def bio_interpretation(a, b):
    pair = f"{norm(a)} {norm(b)}"

    if any(x in pair for x in ["nitrate", "nitrogen", "nutrient"]) and "sxt" in pair:
        return "Nutrient availability may regulate core sxt genes involved in saxitoxin biosynthesis."

    if any(x in pair for x in ["phosphate", "phosphorus"]) and "sxt" in pair:
        return "Phosphorus availability may influence sxt gene-associated toxin biosynthesis."

    if any(x in pair for x in ["light", "salinity", "temperature", "warming"]) and "sxt" in pair:
        return "Environmental conditions may modulate expression or activity of saxitoxin biosynthesis genes."

    if "bloom" in pair and any(x in pair for x in ["biosynthesis", "toxin", "sxt"]):
        return "Bloom dynamics may be linked to activation or ecological expression of saxitoxin biosynthesis."

    if "alexandrium" in pair and "sxt" in pair:
        return "Species-specific association between Alexandrium taxa and saxitoxin biosynthesis genes."

    if "alexandrium" in pair and any(x in pair for x in ["gtx", "neosaxitoxin", "saxitoxin"]):
        return "Species-specific association between Alexandrium taxa and toxin analog profiles."

    if "pyrodinium_bahamense" in pair and any(x in pair for x in ["gtx", "neosaxitoxin", "saxitoxin"]):
        return "Species-specific association between Pyrodinium bahamense and toxin analog profiles."

    if "sxt" in pair and any(x in pair for x in ["biosynthesis", "production", "expression", "regulation"]):
        return "Mechanistic coupling between sxt genes and toxin biosynthesis or regulation."

    if any(x in pair for x in ["gtx", "neosaxitoxin", "saxitoxin"]) and any(x in pair for x in ["biosynthesis", "production", "expression"]):
        return "Toxin analog production may be linked to biosynthetic or transcriptional regulation."

    return "Biologically plausible STX-related relationship requiring targeted experimental or literature validation."

print("Loading files...")
df = pd.read_csv(CANDIDATES).fillna("")
post = pd.read_csv(POST_EDGES).fillna("")

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])
model.fit(X_train, y_train)

df["AI_Probability"] = model.predict_proba(X)[:, 1]
df["is_actionable"] = df.apply(lambda r: actionable(r["source"], r["target"]), axis=1)

df = df[df["is_actionable"]].copy()

post["pair"] = post.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)

paper_col = None
for c in ["document_id", "paper_id", "pmid", "doi", "title", "source_file"]:
    if c in post.columns:
        paper_col = c
        break

year_col = None
for c in ["year", "Year", "publication_year"]:
    if c in post.columns:
        year_col = c
        break

evidence = []
for p, g in post.groupby("pair"):
    n = g[paper_col].astype(str).nunique() if paper_col else len(g)
    if year_col:
        yrs = pd.to_numeric(g[year_col], errors="coerce").dropna()
        year = int(yrs.min()) if len(yrs) else ""
    else:
        year = ""
    evidence.append({"pair": p, "Supporting_Post2015_Papers": n, "Earliest_Validation_Year": year})

evidence = pd.DataFrame(evidence)

df["pair"] = df.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)
df = df.merge(evidence, on="pair", how="left")
df["Supporting_Post2015_Papers"] = df["Supporting_Post2015_Papers"].fillna(0).astype(int)
df["Earliest_Validation_Year"] = df["Earliest_Validation_Year"].fillna("")

df["Hypothesis"] = df["source"].apply(clean_label) + " ↔ " + df["target"].apply(clean_label)
df["Biological_Interpretation"] = df.apply(lambda r: bio_interpretation(r["source"], r["target"]), axis=1)

validated = (
    df[df["label"] == 1]
    .sort_values(["AI_Probability", "Supporting_Post2015_Papers"], ascending=False)
    .head(TOP_N)
    .copy()
)
validated.insert(0, "Rank", range(1, len(validated) + 1))
validated["Validation_Status"] = "Validated in post-2015 literature"

unvalidated = (
    df[df["label"] == 0]
    .sort_values("AI_Probability", ascending=False)
    .head(TOP_N)
    .copy()
)
unvalidated.insert(0, "Rank", range(1, len(unvalidated) + 1))
unvalidated["Validation_Status"] = "Unvalidated / candidate future hypothesis"

cols = [
    "Rank",
    "Hypothesis",
    "Validation_Status",
    "AI_Probability",
    "Supporting_Post2015_Papers",
    "Earliest_Validation_Year",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "Biological_Interpretation"
]

validated[cols].to_csv(OUT_VALIDATED, index=False, encoding="utf-8-sig")
unvalidated[cols].to_csv(OUT_UNVALIDATED, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(OUT_VALIDATED)
print(OUT_UNVALIDATED)

print("\nTop actionable validated hypotheses:")
print(validated[cols].head(20).to_string(index=False))

print("\nTop actionable unvalidated hypotheses:")
print(unvalidated[cols].head(20).to_string(index=False))
