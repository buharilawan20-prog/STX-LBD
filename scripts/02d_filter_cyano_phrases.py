from pathlib import Path
import pandas as pd


INPUT_FILE = "data/processed/cyano_ngram_phrases.csv"
OUTPUT_FILE = "data/processed/cyano_ngram_phrases_filtered.csv"
OUTPUT_ENTITY_FILE = "data/processed/entities_cyano_ngram_phrases_filtered.csv"


BAD_PHRASES = {
    "this study",
    "have been",
    "has been",
    "there are",
    "there is",
    "were used",
    "was used",
    "can be",
    "may be",
    "could be",
    "these results",
    "our results",
    "previous studies",
    "present study",
    "human health",
    "liquid chromatography",
    "mass spectrometry",
}


BIO_KEYWORDS = [
    "saxitoxin",
    "stx",
    "sxt",
    "paralytic",
    "shellfish",
    "toxin",
    "toxins",
    "cyanobacteria",
    "cyanobacterial",
    "cyanotoxin",
    "cyanotoxins",
    "blooms",
    "biosynthesis",
    "gene",
    "genes",
    "cluster",
    "Cylindrospermopsis".lower(),
    "Raphidiopsis".lower(),
    "Dolichospermum".lower(),
    "Aphanizomenon".lower(),
    "Microseira".lower(),
    "Lyngbya".lower(),
    "nitrogen",
    "phosphate",
    "temperature",
    "salinity",
    "pH".lower(),
    "freshwater",
    "drinking water",
    "harmful algal",
]


def keep_phrase(phrase):
    p = str(phrase).lower().strip()

    if p in BAD_PHRASES:
        return False

    if len(p.split()) < 2:
        return False

    if any(k.lower() in p for k in BIO_KEYWORDS):
        return True

    return False


def infer_phrase_type(phrase):
    p = str(phrase).lower()

    if any(x in p for x in ["sxt", "gene", "cluster"]):
        return "PHRASE_GENE_MECHANISM"

    if any(x in p for x in ["biosynthesis", "pathway"]):
        return "PHRASE_BIOSYNTHESIS"

    if any(x in p for x in ["saxitoxin", "stx", "toxin", "toxins", "paralytic", "shellfish"]):
        return "PHRASE_TOXIN_PHENOTYPE"

    if any(x in p for x in ["cyanobacteria", "cyanobacterial", "raphidiopsis", "cylindrospermopsis", "aphanizomenon", "dolichospermum"]):
        return "PHRASE_TAXON"

    if any(x in p for x in ["nitrogen", "phosphate", "temperature", "salinity", "freshwater", "drinking water", "ph"]):
        return "PHRASE_ENVIRONMENT"

    return "PHRASE_OTHER"


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    df = df[df["phrase"].apply(keep_phrase)].copy()
    df["phrase_type"] = df["phrase"].apply(infer_phrase_type)

    df = df.sort_values("count", ascending=False)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    ent = pd.DataFrame({
        "paper_id": "CYANO_PHRASE",
        "year": "",
        "group": "cyano_train",
        "source": "ngram_phrase",
        "entity_text": df["phrase"],
        "entity_normalized": df["phrase"],
        "entity_type": df["phrase_type"],
        "relation_hint": "",
        "start": 0,
        "end": 0,
        "context": "",
    })

    ent.to_csv(OUTPUT_ENTITY_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved filtered phrases: {OUTPUT_FILE}")
    print(f"Saved filtered phrase entities: {OUTPUT_ENTITY_FILE}")
    print(f"Remaining phrases: {len(df)}")
    print()
    print("Top filtered phrases:")
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
