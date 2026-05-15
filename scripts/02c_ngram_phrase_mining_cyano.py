from pathlib import Path
import pandas as pd
import re
from collections import Counter


INPUT_FILES = ["data/processed/cyano_all_clean.csv"]

OUTPUT_PHRASES = "data/processed/cyano_ngram_phrases.csv"
OUTPUT_ENTITIES = "data/processed/entities_cyano_ngram_phrases.csv"

MIN_GLOBAL_SUPPORT = 5


STOPWORDS = set([
    "the","and","of","in","to","a","for","on","with","is","by","as","an","at","from"
])


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def generate_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def main():
    texts = []

    for file in INPUT_FILES:
        df = pd.read_csv(file, encoding="utf-8-sig")
        for _, row in df.iterrows():
            texts.append(str(row.get("title", "")))
            texts.append(str(row.get("abstract", "")))

    counter = Counter()

    for text in texts:
        tokens = tokenize(text)

        for n in [2, 3, 4]:
            ngrams = generate_ngrams(tokens, n)
            counter.update(ngrams)

    # Filter by support
    phrases = [
        (phrase, count)
        for phrase, count in counter.items()
        if count >= MIN_GLOBAL_SUPPORT
    ]

    phrases_df = pd.DataFrame(phrases, columns=["phrase", "count"])
    phrases_df = phrases_df.sort_values("count", ascending=False)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    phrases_df.to_csv(OUTPUT_PHRASES, index=False, encoding="utf-8-sig")

    # Convert to entity format
    entities = []

    for _, row in phrases_df.iterrows():
        entities.append({
            "entity": row["phrase"],
            "entity_normalized": row["phrase"],
            "entity_type": "PHRASE",
            "count": row["count"],
            "group": "cyano_train"
        })

    entities_df = pd.DataFrame(entities)
    entities_df.to_csv(OUTPUT_ENTITIES, index=False, encoding="utf-8-sig")

    print(f"Saved phrases: {OUTPUT_PHRASES}")
    print(f"Saved phrase entities: {OUTPUT_ENTITIES}")
    print(f"Total phrases: {len(phrases_df)}")


if __name__ == "__main__":
    main()
