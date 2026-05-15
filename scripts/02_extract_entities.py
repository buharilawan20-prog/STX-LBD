from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


INPUT_FILES = [
    "data/processed/dino_pre2016.csv",
    "data/processed/dino_post2015.csv",
]

OUTPUT_FILE = "data/processed/entities_all.csv"


ENTITY_DICTIONARY = {
    "GENE": [
        "sxtA", "sxtA1", "sxtA2", "sxtA3", "sxtA4",
        "sxtB", "sxtC", "sxtD", "sxtG", "sxtH", "sxtI",
        "sxtN", "sxtO", "sxtP", "sxtQ", "sxtS", "sxtT",
        "sxtU", "sxtV", "sxtW", "sxtX", "sxtY", "sxtZ",
        "PKS", "polyketide synthase",
        "FAS", "fatty acid synthase",
    ],

    "SPECIES": [
        "Gymnodinium catenatum",
        "Gymnodinium impudicum",
        "Gymnodinium smaydae",
        "Alexandrium catenella",
        "Alexandrium pacificum",
        "Alexandrium tamarense",
        "Alexandrium minutum",
        "Alexandrium fundyense",
        "Alexandrium ostenfeldii",
        "Pyrodinium bahamense",
        "Centrodinium punctatum",
        "Gonyaulax spinifera",
    ],

    "TOXIN": [
        "saxitoxin",
        "STX",
        "paralytic shellfish toxin",
        "paralytic shellfish toxins",
        "PST",
        "PSTs",
        "gonyautoxin",
        "GTX",
        "neosaxitoxin",
        "neoSTX",
        "decarbamoylsaxitoxin",
        "dcSTX",
    ],

    "ENV_FACTOR": [
        "temperature",
        "salinity",
        "light",
        "irradiance",
        "nitrogen",
        "nitrate",
        "ammonium",
        "phosphate",
        "phosphorus",
        "silicate",
        "nutrient limitation",
        "nutrient stress",
        "pH",
        "CO2",
        "carbon dioxide",
        "oxidative stress",
        "hypoxia",
    ],

    "PROCESS": [
        "biosynthesis",
        "gene expression",
        "transcription",
        "transcriptome",
        "transcriptomics",
        "regulation",
        "evolution",
        "phylogeny",
        "phylogenetic",
        "homolog",
        "homology",
        "horizontal gene transfer",
        "HGT",
        "gene loss",
        "gene duplication",
        "enzyme",
        "enzymatic activity",
        "metabolic pathway",
    ],
}


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_pattern(term: str) -> re.Pattern:
    """
    Build case-insensitive regex pattern.
    Uses word boundaries where possible.
    """
    escaped = re.escape(term)

    # Allow periods/spaces for abbreviated species names later if needed
    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_entities_from_text(text: str, paper_id: str, year: int, group: str, source: str):
    rows = []

    for entity_type, terms in ENTITY_DICTIONARY.items():
        for term in terms:
            pattern = make_pattern(term)

            for match in pattern.finditer(text):
                rows.append({
                    "paper_id": paper_id,
                    "year": year,
                    "group": group,
                    "source": source,
                    "entity_text": match.group(0),
                    "canonical_hint": term,
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                })

    return rows


def main():
    all_rows = []

    for file in INPUT_FILES:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {file}")

        df = pd.read_csv(path, encoding="utf-8-sig")

        required = ["paper_id", "title", "abstract", "year", "group"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file} missing columns: {missing}")

        for _, row in df.iterrows():
            paper_id = str(row["paper_id"])
            year = int(row["year"])
            group = str(row["group"])

            title = clean_text(row.get("title", ""))
            abstract = clean_text(row.get("abstract", ""))

            all_rows.extend(
                extract_entities_from_text(title, paper_id, year, group, "title")
            )
            all_rows.extend(
                extract_entities_from_text(abstract, paper_id, year, group, "abstract")
            )

    ent_df = pd.DataFrame(all_rows)

    if ent_df.empty:
        print("No entities found. Check dictionary terms and corpus text.")
        return

    ent_df = ent_df.drop_duplicates(
        subset=["paper_id", "entity_text", "entity_type", "start", "end", "source"]
    )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ent_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved entities: {output_path}")
    print(f"Total extracted entities: {len(ent_df)}")
    print()
    print("Entity counts by type:")
    print(ent_df["entity_type"].value_counts())
    print()
    print("Entity counts by group:")
    print(ent_df["group"].value_counts())


if __name__ == "__main__":
    main()
