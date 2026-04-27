from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_FILE = "data/processed/entities_all.csv"
OUTPUT_FILE = "data/processed/entities_normalized.csv"


NORMALIZATION_MAP = {
    # TOXINS
    "stx": "saxitoxin",
    "saxitoxin": "saxitoxin",
    "pst": "paralytic shellfish toxins",
    "psts": "paralytic shellfish toxins",
    "paralytic shellfish toxin": "paralytic shellfish toxins",
    "paralytic shellfish toxins": "paralytic shellfish toxins",
    "gtx": "gonyautoxin",
    "gonyautoxin": "gonyautoxin",
    "neostx": "neosaxitoxin",
    "neosaxitoxin": "neosaxitoxin",
    "dcstx": "decarbamoylsaxitoxin",
    "decarbamoylsaxitoxin": "decarbamoylsaxitoxin",

    # GENES
    "sxta": "sxtA",
    "sxta1": "sxtA1",
    "sxta2": "sxtA2",
    "sxta3": "sxtA3",
    "sxta4": "sxtA4",
    "sxtb": "sxtB",
    "sxtc": "sxtC",
    "sxtd": "sxtD",
    "sxtg": "sxtG",
    "sxth": "sxtH",
    "sxti": "sxtI",
    "sxtn": "sxtN",
    "sxto": "sxtO",
    "sxtp": "sxtP",
    "sxtq": "sxtQ",
    "sxts": "sxtS",
    "sxtt": "sxtT",
    "sxtu": "sxtU",
    "sxtv": "sxtV",
    "sxtw": "sxtW",
    "sxtx": "sxtX",
    "sxty": "sxtY",
    "sxtz": "sxtZ",
    "pks": "PKS",
    "polyketide synthase": "PKS",
    "fas": "FAS",
    "fatty acid synthase": "FAS",

    # SPECIES
    "gymnodinium catenatum": "Gymnodinium catenatum",
    "gymnodinium impudicum": "Gymnodinium impudicum",
    "gymnodinium smaydae": "Gymnodinium smaydae",
    "alexandrium catenella": "Alexandrium catenella",
    "alexandrium pacificum": "Alexandrium pacificum",
    "alexandrium tamarense": "Alexandrium tamarense",
    "alexandrium minutum": "Alexandrium minutum",
    "alexandrium fundyense": "Alexandrium fundyense",
    "alexandrium ostenfeldii": "Alexandrium ostenfeldii",
    "pyrodinium bahamense": "Pyrodinium bahamense",
    "centrodinium punctatum": "Centrodinium punctatum",
    "gonyaulax spinifera": "Gonyaulax spinifera",

    # ENVIRONMENT
    "temperature": "temperature",
    "salinity": "salinity",
    "light": "light",
    "irradiance": "light",
    "nitrogen": "nitrogen",
    "nitrate": "nitrate",
    "ammonium": "ammonium",
    "phosphate": "phosphate",
    "phosphorus": "phosphorus",
    "silicate": "silicate",
    "nutrient limitation": "nutrient limitation",
    "nutrient stress": "nutrient stress",
    "ph": "pH",
    "co2": "CO2",
    "carbon dioxide": "CO2",
    "oxidative stress": "oxidative stress",
    "hypoxia": "hypoxia",

    # PROCESS
    "biosynthesis": "biosynthesis",
    "gene expression": "gene expression",
    "transcription": "transcription",
    "transcriptome": "transcriptome",
    "transcriptomics": "transcriptomics",
    "regulation": "regulation",
    "evolution": "evolution",
    "phylogeny": "phylogeny",
    "phylogenetic": "phylogeny",
    "homolog": "homology",
    "homology": "homology",
    "horizontal gene transfer": "horizontal gene transfer",
    "hgt": "horizontal gene transfer",
    "gene loss": "gene loss",
    "gene duplication": "gene duplication",
    "enzyme": "enzyme",
    "enzymatic activity": "enzymatic activity",
    "metabolic pathway": "metabolic pathway",
}


def clean_key(text: object) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip().lower().replace("-", "").replace("_", "")


def normalize_entity(row: pd.Series) -> str:
    raw = row.get("entity_text", "")
    hint = row.get("canonical_hint", "")

    raw_key = clean_key(raw)
    hint_key = clean_key(hint)

    if raw_key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[raw_key]

    if hint_key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[hint_key]

    return str(raw).strip()


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required = ["paper_id", "year", "group", "entity_text", "canonical_hint", "entity_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["entity_normalized"] = df.apply(normalize_entity, axis=1)

    # Remove empty normalized entities
    df = df[df["entity_normalized"].astype(str).str.strip() != ""]

    # Drop exact duplicate entity mentions
    df = df.drop_duplicates(
        subset=[
            "paper_id",
            "year",
            "group",
            "entity_type",
            "entity_normalized",
            "source",
            "start",
            "end",
        ]
    )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved normalized entities: {output_path}")
    print(f"Total normalized entity mentions: {len(df)}")
    print()
    print("Top normalized entities:")
    print(df["entity_normalized"].value_counts().head(30))
    print()
    print("Entity counts by type:")
    print(df["entity_type"].value_counts())
    print()
    print("Entity counts by group:")
    print(df["group"].value_counts())


if __name__ == "__main__":
    main()
