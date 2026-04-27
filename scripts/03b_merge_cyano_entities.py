from pathlib import Path
import pandas as pd


ADVANCED_ENTITY_FILE = "data/processed/entities_cyano_advanced.csv"
NGRAM_ENTITY_FILE = "data/processed/entities_cyano_ngram_phrases_filtered.csv"

OUTPUT_FILE = "data/processed/entities_cyano_semantic_merged.csv"


def main():
    adv = pd.read_csv(ADVANCED_ENTITY_FILE, encoding="utf-8-sig")
    ngram = pd.read_csv(NGRAM_ENTITY_FILE, encoding="utf-8-sig")

    required = [
        "paper_id",
        "year",
        "group",
        "source",
        "entity_text",
        "entity_normalized",
        "entity_type",
        "relation_hint",
        "start",
        "end",
        "context",
    ]

    for name, df in [("advanced", adv), ("ngram", ngram)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file missing columns: {missing}")

    merged = pd.concat([adv[required], ngram[required]], ignore_index=True)

    merged["entity_normalized"] = (
        merged["entity_normalized"]
        .astype(str)
        .str.strip()
    )

    merged = merged[merged["entity_normalized"] != ""]
    merged = merged[merged["entity_normalized"].str.lower() != "nan"]

    merged = merged.drop_duplicates(
        subset=[
            "paper_id",
            "group",
            "entity_type",
            "entity_normalized",
            "source",
            "start",
            "end",
        ]
    )

    output = Path(OUTPUT_FILE)
    output.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Saved merged cyano semantic entities: {output}")
    print(f"Total merged records: {len(merged)}")
    print()
    print("Entity counts by type:")
    print(merged["entity_type"].value_counts().head(40))
    print()
    print("Top normalized entities:")
    print(merged["entity_normalized"].value_counts().head(40))


if __name__ == "__main__":
    main()
