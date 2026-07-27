import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

CANDIDATES = BASE / "FINAL_WORKSPACE/strict_validation_v2/strict_global_candidates.csv"

OUT = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT_VALIDATED = OUT / "top_validated_hypotheses_with_interpretation.csv"
OUT_UNVALIDATED = OUT / "top_unvalidated_hypotheses_with_interpretation.csv"

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "degree_u",
    "degree_v"
]

TOP_N = 30

def clean_label(x):
    x = str(x).replace("_", " ")
    repl = {
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth": "sxtH",
        "sxts": "sxtS",
        "gtx": "GTX",
        "neosaxitoxin": "neoSTX",
        "saxitoxin biosynthesis": "STX biosynthesis",
        "toxin biosynthesis": "toxin biosynthesis",
        "toxin production": "toxin production",
        "alexandrium catenella": "Alexandrium catenella",
        "alexandrium pacificum": "Alexandrium pacificum",
        "alexandrium minutum": "Alexandrium minutum",
        "alexandrium fundyense": "Alexandrium fundyense",
        "gymnodinium catenatum": "Gymnodinium catenatum",
        "pyrodinium bahamense": "Pyrodinium bahamense"
    }
    return repl.get(x, x)

def biological_interpretation(a, b):
    pair = f"{a} {b}".lower()

    if any(x in pair for x in ["nitrate", "nitrogen", "nutrient"]) and "sxt" in pair:
        return "Suggests nutrient-mediated regulation of sxt genes and saxitoxin biosynthesis."

    if any(x in pair for x in ["salinity", "light", "temperature", "warming"]) and "sxt" in pair:
        return "Suggests environmental modulation of toxin biosynthesis gene dynamics."

    if "gtx" in pair and any(x in pair for x in ["biosynthesis", "expression", "production"]):
        return "Links gonyautoxin-related toxin profiles with biosynthetic or regulatory processes."

    if "sxt" in pair and any(x in pair for x in ["biosynthesis", "production", "regulation", "expression"]):
        return "Supports mechanistic coupling between sxt genes and toxin biosynthesis/regulation."

    if "alexandrium" in pair and "sxt" in pair:
        return "Indicates taxon-specific association between Alexandrium species and sxt gene repertoire."

    if "alexandrium" in pair and any(x in pair for x in ["gtx", "neosaxitoxin", "saxitoxin"]):
        return "Represents species-level association with saxitoxin analog production."

    if any(x in pair for x in ["gene_loss", "evolution", "phylogeny"]) and "sxt" in pair:
        return "Suggests evolutionary restructuring, retention, or loss of saxitoxin-related genes."

    if any(x in pair for x in ["bloom", "harmful_algal_bloom"]) and any(x in pair for x in ["sxt", "gtx", "toxin"]):
        return "Suggests bloom-associated toxin gene or toxin phenotype dynamics."

    return "Indicates a predicted semantic relationship within dinoflagellate saxitoxin biology."

df = pd.read_csv(CANDIDATES).fillna("")

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

df["AI_Probability"] = model.predict_proba(X)[:, 1]
df["Source_Label"] = df["source"].apply(clean_label)
df["Target_Label"] = df["target"].apply(clean_label)
df["Hypothesis"] = df["Source_Label"] + " ↔ " + df["Target_Label"]

df["Biological_Interpretation"] = df.apply(
    lambda r: biological_interpretation(r["source"], r["target"]),
    axis=1
)

validated = (
    df[df["label"] == 1]
    .sort_values("AI_Probability", ascending=False)
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

print("\nTop validated hypotheses:")
print(validated[cols].head(15).to_string(index=False))

print("\nTop unvalidated hypotheses:")
print(unvalidated[cols].head(15).to_string(index=False))
