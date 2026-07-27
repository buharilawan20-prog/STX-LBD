import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ======================================================
# PATHS
# ======================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CANDIDATES = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_candidates.csv"
POST_EDGES = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT_TABLE = OUT / "hypothesis_query_demo_table_2.csv"

# ======================================================
# USER-DEFINED EXAMPLE QUERIES
# Add/remove queries here
# ======================================================

QUERIES = [
    ("cyanobacteria", "sxta"),
    ("warming", "saxitoxin_biosynthesis"),
    ("gymnodinium_catenatum", "sxta"),
    ("light", "sxtg"),
    ("salinity", "sxt_genes"),
    ("phosphate", "sxtu"),
    ("gene_expression", "toxin_biosynthesis"),
    ("bloom", "saxitoxin_biosynthesis"),
    ("alexandrium_catenella", "sxtg"),
    ("alexandrium_pacificum", "sxta"),
]

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "degree_u",
    "degree_v"
]

# ======================================================
# HELPERS
# ======================================================

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    return "||".join(sorted([norm(a), norm(b)]))

def clean(x):
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
        "stx": "STX",
        "gtx": "GTX",
        "gonyautoxin": "GTX",
        "neosaxitoxin": "neoSTX",
        "paralytic_shellfish_toxins": "PSTs",
        "saxitoxin_biosynthesis": "STX biosynthesis",
        "stx_biosynthesis": "STX biosynthesis",
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

def classify_term(x):
    x = norm(x)

    if x in {
        "nitrogen", "nitrate", "nutrient", "nutrients",
        "phosphorus", "phosphate", "temperature",
        "warming", "salinity", "light"
    }:
        return "ENV"

    if x.startswith("sxt") or x in {"sxt_genes"}:
        return "GENE"

    if x in {
        "saxitoxin", "stx", "gtx", "gonyautoxin",
        "neosaxitoxin", "toxin_profile", "toxin_profiles",
        "toxin_production"
    }:
        return "TOXIN"

    if "biosynthesis" in x or x in {
        "growth", "regulation", "expression",
        "gene_expression", "toxin_biosynthesis"
    }:
        return "MECHANISM"

    if x in {"evolution", "phylogeny", "phylogenetic"}:
        return "EVOLUTION"

    if x.startswith("alexandrium") or x.startswith("gymnodinium") or x.startswith("pyrodinium"):
        return "TAXON"

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

    if cats == {"GENE", "MECHANISM"}:
        return "GENE_MECHANISM_HYPOTHESIS"

    if cats == {"MECHANISM", "TOXIN"}:
        return "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS"

    if "TAXON" in cats and "GENE" in cats:
        return "TAXON_GENE_HYPOTHESIS"

    if "TAXON" in cats and "TOXIN" in cats:
        return "TAXON_TOXIN_HYPOTHESIS"

    if "EVOLUTION" in cats and ("GENE" in cats or "MECHANISM" in cats):
        return "EVOLUTION_GENE_MECHANISM_HYPOTHESIS"

    return "OTHER_STX_HYPOTHESIS"

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

    if rel_type == "GENE_MECHANISM_HYPOTHESIS":
        return "Suggests mechanistic coupling between STX-related genes and biosynthetic or regulatory processes."

    if rel_type == "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS":
        return "Highlights biosynthetic or regulatory activity as a mechanistic driver linking STX pathway activity with toxin phenotype."

    if rel_type == "TAXON_GENE_HYPOTHESIS":
        taxon = clean(a) if classify_term(a) == "TAXON" else clean(b)
        gene = clean(a) if classify_term(a) == "GENE" else clean(b)
        return f"Suggests a species- or lineage-specific association between {taxon} and {gene}."

    if rel_type == "TAXON_TOXIN_HYPOTHESIS":
        taxon = clean(a) if classify_term(a) == "TAXON" else clean(b)
        return f"Suggests {taxon} may be associated with specific STX or toxin analog phenotypes."

    if rel_type == "EVOLUTION_GENE_MECHANISM_HYPOTHESIS":
        return "Suggests evolutionary or phylogenetic processes may shape STX-related genes and biosynthetic pathways."

    return "Represents a biologically plausible STX-related relationship requiring targeted validation."

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(CANDIDATES).fillna("")
post = pd.read_csv(POST_EDGES).fillna("")

# ======================================================
# TRAIN MODEL USING STRICT VALIDATION DATA
# ======================================================

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

df["AI Score"] = model.predict_proba(X)[:, 1]
df["pair"] = df.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)

# ======================================================
# POST-2015 EVIDENCE COUNTS
# ======================================================

post["pair"] = post.apply(lambda r: pair_key(r["source"], r["target"]), axis=1)

def count_docs(x):
    x = str(x).strip()
    if not x or x.lower() in {"nan", "none"}:
        return 0
    docs = [d.strip() for d in x.split(";") if d.strip()]
    return len(set(docs))

if "support_documents" in post.columns:
    evidence = post[["pair", "support_documents"]].copy()
    evidence["Evidence (Papers)"] = evidence["support_documents"].apply(count_docs)
    evidence = evidence.groupby("pair", as_index=False)["Evidence (Papers)"].sum()
else:
    paper_col = None
    for c in ["document_id", "paper_id", "pmid", "doi", "title"]:
        if c in post.columns:
            paper_col = c
            break

    if paper_col:
        evidence = post.groupby("pair")[paper_col].nunique().reset_index()
        evidence.columns = ["pair", "Evidence (Papers)"]
    else:
        evidence = post.groupby("pair").size().reset_index(name="Evidence (Papers)")

# ======================================================
# QUERY FUNCTION
# ======================================================

def query_pair(a, b):
    pk = pair_key(a, b)

    row = df[df["pair"] == pk].copy()

    if row.empty:
        return {
            "Query": f"{clean(a)} - {clean(b)}",
            "Relationship Type": relationship_type(a, b),
            "AI Score": "NA",
            "Evidence (Papers)": 0,
            "Validation Status": "Not in generated candidate space",
            "Biological Interpretation": interpretation(a, b, relationship_type(a, b))
        }

    row = row.iloc[0]

    ev = evidence[evidence["pair"] == pk]
    n_papers = int(ev["Evidence (Papers)"].iloc[0]) if not ev.empty else 0

    rel_type = relationship_type(a, b)

    status = (
        "Recovered in post-2015 literature"
        if int(row["label"]) == 1
        else "Not recovered in post-2015 literature"
    )

    return {
        "Query": f"{clean(a)} - {clean(b)}",
        "Relationship Type": rel_type,
        "AI Score": round(float(row["AI Score"]), 3),
        "Evidence (Papers)": n_papers,
        "Validation Status": status,
        "Biological Interpretation": interpretation(a, b, rel_type)
    }

# ======================================================
# RUN QUERIES
# ======================================================

results = []

for a, b in QUERIES:
    results.append(query_pair(a, b))

out = pd.DataFrame(results)

out.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(OUT_TABLE)

print("\nHypothesis query demonstration:")
print(out.to_string(index=False))
