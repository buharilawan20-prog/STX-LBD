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
OUT_TABLE = OUT / "representative_hypotheses_table.csv"

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "degree_u",
    "degree_v"
]

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    return "||".join(sorted([norm(a), norm(b)]))

def clean(x):
    x = norm(x)
    repl = {
        "sxta": "sxtA",
        "sxta4": "sxtA4",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth": "sxtH",
        "sxts": "sxtS",
        "sxt_genes": "sxt genes",
        "stx_biosynthesis": "STX biosynthesis",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "toxin_biosynthesis": "toxin biosynthesis",
        "toxin_production": "toxin production",
        "toxin_profiles": "toxin profiles",
        "toxin_profile": "toxin profile",
        "gene_expression": "gene expression",
        "genes_involved": "genes involved",
        "gtx": "GTX",
        "gonyautoxin": "GTX",
        "neosaxitoxin": "neoSTX",
        "saxitoxin": "STX"
    }
    return repl.get(x, x.replace("_", " "))

def classify_term(x):
    x = norm(x)

    if x in {"nitrogen", "nitrate", "nutrient", "nutrients", "phosphorus", "phosphate", "temperature", "warming", "salinity", "light"}:
        return "ENV"

    if x.startswith("sxt") or x in {"genes_involved", "genes"}:
        return "GENE"

    if x in {"gtx", "gonyautoxin", "neosaxitoxin", "saxitoxin", "toxin_profile", "toxin_profiles", "toxin_production"}:
        return "TOXIN"

    if "biosynthesis" in x or x in {"growth", "regulation", "expression", "gene_expression"}:
        return "MECHANISM"

    if x in {"evolution", "phylogeny", "phylogenetic"}:
        return "EVOLUTION"

    return "OTHER"

def relationship_type(a, b):
    ca, cb = classify_term(a), classify_term(b)
    cats = {ca, cb}

    if cats == {"ENV", "GENE"}:
        return "ENV_GENE_HYPOTHESIS"

    if cats == {"ENV", "TOXIN"}:
        return "ENV_TOXIN_PHENOTYPE_HYPOTHESIS"

    if cats == {"ENV", "MECHANISM"}:
        return "ENV_MECHANISM_HYPOTHESIS"

    if cats == {"GENE", "TOXIN"}:
        return "GENE_TOXIN_HYPOTHESIS"

    if "EVOLUTION" in cats and ("GENE" in cats or "MECHANISM" in cats):
        return "EVOLUTION_GENE_MECHANISM_HYPOTHESIS"

    if cats == {"MECHANISM", "TOXIN"}:
        return "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS"

    if cats == {"GENE", "MECHANISM"}:
        return "GENE_MECHANISM_HYPOTHESIS"

    return "OTHER_STX_HYPOTHESIS"

def group_name(rel_type):
    if rel_type.startswith("ENV_"):
        return "Environmental regulation"
    if rel_type.startswith("GENE_TOXIN"):
        return "Gene–toxin relationships"
    if rel_type.startswith("EVOLUTION"):
        return "Evolutionary organization"
    if rel_type.startswith("MECHANISM"):
        return "Biosynthetic mechanisms"
    if rel_type.startswith("GENE_MECHANISM"):
        return "Biosynthetic mechanisms"
    return "Other"

def interpretation(a, b, rel_type):
    pair = f"{norm(a)} {norm(b)}"

    if rel_type == "ENV_GENE_HYPOTHESIS":
        env = clean(a) if classify_term(a) == "ENV" else clean(b)
        gene = clean(a) if classify_term(a) == "GENE" else clean(b)
        return f"Suggests {env} may regulate {gene}-associated STX biosynthesis and toxin production dynamics."

    if rel_type == "ENV_TOXIN_PHENOTYPE_HYPOTHESIS":
        env = clean(a) if classify_term(a) == "ENV" else clean(b)
        return f"Suggests {env} availability may influence STX biosynthesis, toxin phenotype, or toxin analog composition."

    if rel_type == "ENV_MECHANISM_HYPOTHESIS":
        env = clean(a) if classify_term(a) == "ENV" else clean(b)
        mech = clean(a) if classify_term(a) == "MECHANISM" else clean(b)
        return f"Indicates {env} may shape {mech} and STX production dynamics."

    if rel_type == "GENE_TOXIN_HYPOTHESIS":
        gene = clean(a) if classify_term(a) == "GENE" else clean(b)
        return f"Supports a role of {gene}-related genes in determining toxin variability, biosynthesis, or toxin analog profiles."

    if rel_type == "EVOLUTION_GENE_MECHANISM_HYPOTHESIS":
        return "Suggests evolutionary or phylogenetic processes may shape the organization, conservation, or diversification of STX-related genes and pathways."

    if rel_type == "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS":
        return "Highlights biosynthetic or regulatory activity as a mechanistic driver linking STX pathway activity with toxin phenotype."

    if rel_type == "GENE_MECHANISM_HYPOTHESIS":
        return "Suggests mechanistic coupling between STX-related genes and biosynthetic or regulatory processes."

    return "Represents a biologically plausible STX-related relationship requiring targeted validation."

# ======================================================
# LOAD AND SCORE
# ======================================================

df = pd.read_csv(CANDIDATES).fillna("")
post = pd.read_csv(POST_EDGES).fillna("")

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
df["AI Score"] = model.predict_proba(X)[:, 1]

# ======================================================
# EVIDENCE COUNTS
# ======================================================

post["pair"] = post.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)

def count_support_documents(x):
    x = str(x).strip()

    if not x or x.lower() in ["nan", "none"]:
        return 0

    docs = [d.strip() for d in x.split(";") if d.strip()]
    return len(set(docs))

if "support_documents" in post.columns:
    evidence = post[["pair", "support_documents"]].copy()
    evidence["Evidence_Papers"] = evidence["support_documents"].apply(count_support_documents)

    evidence = (
        evidence.groupby("pair", as_index=False)["Evidence_Papers"]
        .sum()
    )

else:
    paper_col = None
    for c in ["document_id", "paper_id", "pmid", "doi", "title"]:
        if c in post.columns:
            paper_col = c
            break

    if paper_col:
        evidence = post.groupby("pair")[paper_col].nunique().reset_index()
        evidence.columns = ["pair", "Evidence_Papers"]
    else:
        evidence = post.groupby("pair").size().reset_index(name="Evidence_Papers")

df["pair"] = df.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)

# Remove old evidence columns if they exist
for c in ["Evidence (Papers)", "Evidence_Papers", "Evidence (Papers)_x", "Evidence (Papers)_y"]:
    if c in df.columns:
        df = df.drop(columns=[c])

df = df.merge(evidence, on="pair", how="left")

df["Evidence_Papers"] = df["Evidence_Papers"].fillna(0).astype(int)
df["Evidence (Papers)"] = df["Evidence_Papers"]


# ======================================================
# FORMAT
# ======================================================

df["Relationship Type"] = df.apply(
    lambda r: relationship_type(r["source"], r["target"]),
    axis=1
)

df["Group"] = df["Relationship Type"].apply(group_name)

df["Representative Hypothesis"] = df.apply(
    lambda r: f"{clean(r['source'])} - {clean(r['target'])}",
    axis=1
)

df["Biological Interpretation"] = df.apply(
    lambda r: interpretation(r["source"], r["target"], r["Relationship Type"]),
    axis=1
)

# keep only meaningful groups and validated future positives
df = df[
    (df["label"] == 1) &
    (df["Relationship Type"] != "OTHER_STX_HYPOTHESIS")
].copy()

# select top examples per group
wanted_groups = [
    "Environmental regulation",
    "Gene–toxin relationships",
    "Evolutionary organization",
    "Biosynthetic mechanisms"
]

out_rows = []

for group in wanted_groups:
    g = df[df["Group"] == group].sort_values("AI Score", ascending=False).head(3)
    for _, r in g.iterrows():
        out_rows.append(r)

out = pd.DataFrame(out_rows)

out["AI Score"] = out["AI Score"].round(3)

out = out[
    [
        "Group",
        "Representative Hypothesis",
        "Relationship Type",
        "AI Score",
        "Evidence (Papers)",
        "Biological Interpretation"
    ]
]

out.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(OUT_TABLE)

print("\nRepresentative hypothesis table:")
print(out.to_string(index=False))
