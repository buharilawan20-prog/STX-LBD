from pathlib import Path
import re
import pandas as pd


INPUT_FILE = "cyano_master.csv"
OUTPUT_FILE = "data/processed/cyano_all_clean.csv"


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = ["Title", "Abstract", "Year", "Paper_Id", "Domain", "Group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.rename(columns={
        "Collection": "collection",
        "Title": "title",
        "Abstract": "abstract",
        "Journal": "journal",
        "Year": "year",
        "Paper_Id": "paper_id",
        "Domain": "domain",
        "Group": "group",
    })

    df["title"] = df["title"].apply(clean_text)
    df["abstract"] = df["abstract"].apply(clean_text)
    df["journal"] = df.get("journal", "").apply(clean_text) if "journal" in df.columns else ""

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["domain"] = "cyanobacteria"
    df["group"] = "cyano_train"

    df = df[(df["title"] != "") | (df["abstract"] != "")].copy()

    output = Path(OUTPUT_FILE)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Saved: {output}")
    print(f"Total cyanobacteria records: {len(df)}")
    print("Year range:", df["year"].min(), "-", df["year"].max())


if __name__ == "__main__":
    main()
