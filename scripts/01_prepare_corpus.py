from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_FILE = "corpus_master.csv"
OUTPUT_DIR = "data/processed"


def clean_text(text: object) -> str:
    """Basic text cleaning for title/abstract fields."""
    if pd.isna(text):
        return ""
    text = str(text)

    # Remove simple HTML tags like <i>...</i>
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a consistent internal format."""
    rename_map = {
        "Collection": "collection",
        "Title": "title",
        "Abstract": "abstract",
        "Journal": "journal",
        "Year": "year",
        "Paper_Id": "paper_id",
        "Paper_ID": "paper_id",
        "PaperId": "paper_id",
        "Group": "group",
        "Domain": "domain",
        "DOI": "doi",
        "Tags": "tags",
        "Notes": "notes",
        "Driver": "driver",
        "Mechanism": "mechanism",
        "Outcome": "outcome",
    }

    df = df.rename(columns=rename_map)

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required = ["title", "abstract", "journal", "year", "paper_id", "group", "domain"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize core values."""
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip().str.lower()
    df["domain"] = df["domain"].astype(str).str.strip().str.lower()

    df["title"] = df["title"].apply(clean_text)
    df["abstract"] = df["abstract"].apply(clean_text)
    df["journal"] = df["journal"].apply(clean_text)

    # Remove rows missing critical values
    df = df.dropna(subset=["year"])
    df = df[df["paper_id"] != ""]
    df = df[df["title"] != ""]

    # Convert year back to int after dropping bad rows
    df["year"] = df["year"].astype(int)

    return df


def main() -> None:
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="latin1")

    # 🔥 REMOVE hidden spaces in column names
    df.columns = df.columns.str.strip()
    
    # Remove empty Excel columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = standardize_columns(df)
    validate_required_columns(df)
    df = normalize_values(df)

    # Keep dinoflagellate records only for this first temporal experiment
    dino_df = df[df["domain"] == "dinoflagellate"].copy()

    # Freeze split based on existing group column
    pre2016 = dino_df[dino_df["group"] == "train"].copy()
    post2015 = dino_df[dino_df["group"] == "test"].copy()

    # Save cleaned versions
    dino_df.to_csv(output_dir / "dino_all_clean.csv", index=False, encoding="utf-8-sig")
    pre2016.to_csv(output_dir / "dino_pre2016.csv", index=False, encoding="utf-8-sig")
    post2015.to_csv(output_dir / "dino_post2015.csv", index=False, encoding="utf-8-sig")

    print("Saved files:")
    print(f"  - {output_dir / 'dino_all_clean.csv'}")
    print(f"  - {output_dir / 'dino_pre2016.csv'}")
    print(f"  - {output_dir / 'dino_post2015.csv'}")
    print()
    print(f"Dinoflagellate total: {len(dino_df)}")
    print(f"Train (pre-2016): {len(pre2016)}")
    print(f"Test (post-2015): {len(post2015)}")


if __name__ == "__main__":
    main()
