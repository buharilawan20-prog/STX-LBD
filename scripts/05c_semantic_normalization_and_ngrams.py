import pandas as pd
import re
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/processed/all_split_entities_combined.csv"
OUT_DIR = BASE / "FINAL_WORKSPACE/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "all_split_entities_combined_normalized.csv"
SUMMARY_OUT = OUT_DIR / "entity_normalization_summary.csv"

# ===============================
# NORMALIZATION MAP
# ===============================

NORMALIZATION_MAP = {
    # toxins
    "stx": "saxitoxin",
    "saxitoxins": "saxitoxin",
    "pst": "paralytic_shellfish_toxins",
    "psts": "paralytic_shellfish_toxins",
    "paralytic shellfish toxin": "paralytic_shellfish_toxins",
    "paralytic shellfish toxins": "paralytic_shellfish_toxins",
    "paralytic shellfish poisoning": "paralytic_shellfish_poisoning",

    # sxt genes/domains
    "sxta1": "sxta",
    "sxta2": "sxta",
    "sxta3": "sxta",
    "sxta4": "sxta",
    "sxta domain": "sxta",
    "sxt gene": "sxt_genes",
    "sxt genes": "sxt_genes",
    "saxitoxin biosynthesis gene": "sxt_genes",
    "saxitoxin biosynthesis genes": "sxt_genes",

    # taxa
    "g catenatum": "gymnodinium_catenatum",
    "gymnodinium catenatum": "gymnodinium_catenatum",
    "g smaydae": "gymnodinium_smaydae",
    "gymnodinium smaydae": "gymnodinium_smaydae",
    "g impudicum": "gymnodinium_impudicum",
    "gymnodinium impudicum": "gymnodinium_impudicum",

    "a catenella": "alexandrium_catenella",
    "alexandrium catenella": "alexandrium_catenella",
    "a minutum": "alexandrium_minutum",
    "alexandrium minutum": "alexandrium_minutum",
    "a pacificum": "alexandrium_pacificum",
    "alexandrium pacificum": "alexandrium_pacificum",
    "a tamarense": "alexandrium_tamarense",
    "alexandrium tamarense": "alexandrium_tamarense",
    "a fundyense": "alexandrium_fundyense",
    "alexandrium fundyense": "alexandrium_fundyense",

    "p bahamense": "pyrodinium_bahamense",
    "pyrodinium bahamense": "pyrodinium_bahamense",

    "c punctatum": "centrodinium_punctatum",
    "centrodinium punctatum": "centrodinium_punctatum",

    # processes
    "toxin production": "toxin_production",
    "toxin biosynthesis": "toxin_biosynthesis",
    "saxitoxin biosynthesis": "saxitoxin_biosynthesis",
    "gene expression": "gene_expression",
    "horizontal gene transfer": "horizontal_gene_transfer",
    "gene duplication": "gene_duplication",
    "gene loss": "gene_loss",
    "functional divergence": "functional_divergence",
    "harmful algal bloom": "harmful_algal_bloom",

    # methods
    "lc-ms": "lc_ms",
    "lc-ms/ms": "lc_ms_ms",
    "mass spectrometry": "mass_spectrometry",
    "mouse bioassay": "mouse_bioassay",
    "toxin profiling": "toxin_profiling"
}

# ===============================
# EXTRA N-GRAM PHRASES TO MINE
# ===============================

NGRAM_PHRASES = {
    "harmful algal bloom": ("harmful_algal_bloom", "BIOLOGICAL_PROCESS"),
    "harmful algal blooms": ("harmful_algal_bloom", "BIOLOGICAL_PROCESS"),
    "toxin production": ("toxin_production", "BIOLOGICAL_PROCESS"),
    "toxin biosynthesis": ("toxin_biosynthesis", "BIOLOGICAL_PROCESS"),
    "saxitoxin biosynthesis": ("saxitoxin_biosynthesis", "BIOLOGICAL_PROCESS"),
    "saxitoxin biosynthetic": ("saxitoxin_biosynthesis", "BIOLOGICAL_PROCESS"),
    "gene expression": ("gene_expression", "BIOLOGICAL_PROCESS"),
    "differential expression": ("differential_expression", "BIOLOGICAL_PROCESS"),
    "transcriptional regulation": ("transcriptional_regulation", "BIOLOGICAL_PROCESS"),
    "environmental regulation": ("environmental_regulation", "BIOLOGICAL_PROCESS"),
    "horizontal gene transfer": ("horizontal_gene_transfer", "BIOLOGICAL_PROCESS"),
    "gene duplication": ("gene_duplication", "BIOLOGICAL_PROCESS"),
    "gene loss": ("gene_loss", "BIOLOGICAL_PROCESS"),
    "secondary metabolism": ("secondary_metabolism", "BIOLOGICAL_PROCESS"),
    "secondary metabolites": ("secondary_metabolism", "BIOLOGICAL_PROCESS"),
    "nutrient limitation": ("nutrient_limitation", "ENV_FACTOR"),
    "phosphorus limitation": ("phosphorus_limitation", "ENV_FACTOR"),
    "nitrogen limitation": ("nitrogen_limitation", "ENV_FACTOR"),
    "climate change": ("climate_change", "ENV_FACTOR"),
    "ocean warming": ("ocean_warming", "ENV_FACTOR"),
    "temperature stress": ("temperature_stress", "ENV_FACTOR"),
    "salinity stress": ("salinity_stress", "ENV_FACTOR"),
    "oxidative stress": ("oxidative_stress", "ENV_FACTOR"),
    "bloom dynamics": ("bloom_dynamics", "BIOLOGICAL_PROCESS"),
    "cyst germination": ("cyst_germination", "BIOLOGICAL_PROCESS"),
    "cyst formation": ("cyst_formation", "BIOLOGICAL_PROCESS"),
    "cyst resuspension": ("cyst_resuspension", "BIOLOGICAL_PROCESS"),
    "resting cyst": ("resting_cyst", "BIOLOGICAL_PROCESS"),
    "resting cysts": ("resting_cyst", "BIOLOGICAL_PROCESS"),
    "sxt gene cluster": ("sxt_gene_cluster", "SXT_GENE"),
    "sxt gene clusters": ("sxt_gene_cluster", "SXT_GENE"),
    "sxt genes": ("sxt_genes", "SXT_GENE"),
    "sxta4 domain": ("sxta", "SXT_GENE"),
    "polyketide synthase": ("polyketide_synthase", "BIOLOGICAL_PROCESS"),
    "fatty acid synthase": ("fatty_acid_synthase", "BIOLOGICAL_PROCESS"),
    "liquid chromatography": ("liquid_chromatography", "DETECTION_METHOD"),
    "mass spectrometry": ("mass_spectrometry", "DETECTION_METHOD"),
    "mouse bioassay": ("mouse_bioassay", "DETECTION_METHOD"),
    "toxin profiling": ("toxin_profiling", "DETECTION_METHOD"),
}

# ===============================
# FUNCTIONS
# ===============================

def clean_entity(x):
    x = str(x).lower().strip()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s_\-\/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_entity(x):
    x = clean_entity(x)
    return NORMALIZATION_MAP.get(x, x.replace(" ", "_"))

def normalize_text(x):
    x = str(x).lower()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s\-_\/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def phrase_present(text, phrase):
    text = normalize_text(text)
    phrase = normalize_text(phrase)
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None

# ===============================
# LOAD
# ===============================

df = pd.read_csv(INFILE).fillna("")

for col in ["dataset", "document_id", "year", "taxon_scope", "relevance_class", "entity", "entity_type", "title"]:
    if col not in df.columns:
        df[col] = ""

# ===============================
# NORMALIZE EXISTING ENTITIES
# ===============================

df["entity_original"] = df["entity"]
df["entity_normalized"] = df["entity"].apply(normalize_entity)

# replace entity with normalized form for KG
df["entity"] = df["entity_normalized"]

# ===============================
# ADD N-GRAM PHRASE ENTITIES FROM TITLE
# ===============================

ngram_rows = []

doc_meta = df.groupby("document_id").first().reset_index()

for _, row in doc_meta.iterrows():

    title_text = row.get("title", "")

    for phrase, (norm_phrase, ent_type) in NGRAM_PHRASES.items():

        if phrase_present(title_text, phrase):

            ngram_rows.append({
                "dataset": row["dataset"],
                "document_id": row["document_id"],
                "year": row["year"],
                "taxon_scope": row["taxon_scope"],
                "relevance_class": row["relevance_class"],
                "entity": norm_phrase,
                "entity_type": ent_type,
                "title": row["title"],
                "entity_original": phrase,
                "entity_normalized": norm_phrase,
                "source": "ngram_phrase"
            })

df["source"] = "dictionary_entity"

ngram_df = pd.DataFrame(ngram_rows)

if len(ngram_df) > 0:
    combined = pd.concat([df, ngram_df], ignore_index=True).fillna("")
else:
    combined = df.copy()

# ===============================
# REMOVE DUPLICATE ENTITY MENTIONS
# ===============================

combined = combined.drop_duplicates(
    subset=["dataset", "document_id", "entity", "entity_type"]
).copy()

# ===============================
# SAVE
# ===============================

combined.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

summary = combined.groupby(
    ["dataset", "entity_type"]
).size().reset_index(name="count")

summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

print("\nSaved normalized entity file:")
print(OUTFILE)

print("\nSaved summary:")
print(SUMMARY_OUT)

print("\nEntity mentions after normalization/ngrams:", len(combined))

print("\nEntity types:")
print(combined["entity_type"].value_counts())

print("\nTop normalized entities:")
print(combined["entity"].value_counts().head(30))
