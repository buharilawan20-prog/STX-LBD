from pathlib import Path
import pandas as pd


ADVANCED_ENTITY_FILE = "data/processed/entities_advanced.csv"
NGRAM_ENTITY_FILE = "data/processed/entities_ngram_phrases.csv"

OUTPUT_FILE = "data/processed/entities_semantic_merged.csv"


def main():
    adv_path = Path(ADVANCED_ENTITY_FILE)
    ngram_path = Path(NGRAM_ENTITY_FILE)

    if not adv_path.exists():
        raise FileNotFoundError(f"Missing file: {ADVANCED_ENTITY_FILE}")

    if not ngram_path.exists():
        raise FileNotFoundError(f"Missing file: {NGRAM_ENTITY_FILE}")

    adv = pd.read_csv(adv_path, encoding="utf-8-sig")
    ngram = pd.read_csv(ngram_path, encoding="utf-8-sig")

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

    for df_name, df in [("advanced", adv), ("ngram", ngram)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} file missing columns: {missing}")

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

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved merged semantic entities: {output_path}")
    print(f"Total merged entity records: {len(merged)}")
    print()
    print("Entity counts by type:")
    print(merged["entity_type"].value_counts().head(50))
    print()
    print("Top normalized entities:")
    print(merged["entity_normalized"].value_counts().head(50))


if __name__ == "__main__":
    main()
