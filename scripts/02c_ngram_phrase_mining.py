from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd


INPUT_FILES = [
    "data/processed/dino_pre2016.csv",
    "data/processed/dino_post2015.csv",
]

OUTPUT_FILE = "data/processed/ngram_phrases.csv"
OUTPUT_ENTITY_FILE = "data/processed/entities_ngram_phrases.csv"


MIN_GLOBAL_SUPPORT = 3
NGRAM_MIN = 2
NGRAM_MAX = 4


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "from",
    "by", "with", "as", "at", "into", "during", "between", "among", "than",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "we", "our", "their", "its", "it", "which", "using",
    "used", "use", "study", "studies", "result", "results", "analysis",
    "analyses", "showed", "revealed", "suggest", "suggests", "indicate",
    "indicates", "including", "based", "within", "across", "however",
}


BIO_KEYWORDS = {
    "sxt", "sxta", "sxta1", "sxta4", "sxtg", "sxtb", "sxtu",
    "saxitoxin", "stx", "pst", "psts", "toxin", "toxins",
    "biosynthesis", "gene", "genes", "homolog", "homologs",
    "expression", "transcriptional", "regulation", "regulated",
    "evolution", "evolutionary", "phylogeny", "phylogenetic",
    "clade", "divergence", "conserved", "acquisition", "hgt",
    "temperature", "salinity", "nitrogen", "nitrate", "phosphate",
    "phosphorus", "nutrient", "limitation", "stress",
    "alexandrium", "gymnodinium", "pyrodinium", "cyanobacteria",
    "dinoflagellate", "dinoflagellates",
    "pks", "fas", "polyketide", "synthase",
    "presence", "absence", "loss", "duplication", "retained",
    "toxigenic", "toxic", "non", "nontoxic", "non-toxic",
}


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> list[str]:
    text = text.lower()

    # Preserve gene-like terms and hyphenated terms reasonably
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", text)

    cleaned = []
    for tok in tokens:
        tok = tok.strip("-").lower()
        if not tok:
            continue
        cleaned.append(tok)

    return cleaned


def is_interesting_ngram(tokens: list[str]) -> bool:
    if len(tokens) < NGRAM_MIN:
        return False

    if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
        return False

    if all(t in STOPWORDS for t in tokens):
        return False

    phrase = " ".join(tokens)

    # Avoid very generic phrases
    bad_fragments = [
        "this study",
        "our results",
        "present study",
        "previous studies",
        "in this",
        "of the",
        "and the",
    ]
    if any(bad in phrase for bad in bad_fragments):
        return False

    # Keep phrases with at least one biological keyword
    if not any(t.replace("-", "") in BIO_KEYWORDS or t in BIO_KEYWORDS for t in tokens):
        return False

    return True


def generate_ngrams(tokens: list[str]) -> list[str]:
    phrases = []

    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(0, len(tokens) - n + 1):
            gram_tokens = tokens[i:i + n]

            if is_interesting_ngram(gram_tokens):
                phrases.append(" ".join(gram_tokens))

    return phrases


def infer_phrase_type(phrase: str) -> str:
    p = phrase.lower()

    if any(x in p for x in ["sxt", "gene", "homolog", "domain"]):
        return "PHRASE_GENE_MECHANISM"

    if any(x in p for x in ["expression", "transcription", "regulated", "regulation"]):
        return "PHRASE_REGULATION"

    if any(x in p for x in ["evolution", "phylogen", "clade", "divergence", "hgt", "acquisition", "conserved"]):
        return "PHRASE_EVOLUTION"

    if any(x in p for x in ["temperature", "salinity", "nitrogen", "nitrate", "phosphate", "nutrient", "stress"]):
        return "PHRASE_ENVIRONMENT"

    if any(x in p for x in ["toxin", "saxitoxin", "stx", "pst", "toxigenic", "toxic"]):
        return "PHRASE_TOXIN_PHENOTYPE"

    if any(x in p for x in ["biosynthesis", "pathway", "enzyme", "activity", "synthase"]):
        return "PHRASE_BIOSYNTHESIS"

    if any(x in p for x in ["alexandrium", "gymnodinium", "pyrodinium", "dinoflagellate", "cyanobacteria"]):
        return "PHRASE_TAXON"

    return "PHRASE_OTHER"


def main():
    phrase_counter = Counter()
    phrase_papers = defaultdict(set)
    phrase_years = defaultdict(set)
    phrase_groups = defaultdict(set)
    phrase_contexts = defaultdict(list)

    paper_texts = []

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

            text = clean_text(str(row.get("title", "")) + ". " + str(row.get("abstract", "")))

            paper_texts.append({
                "paper_id": paper_id,
                "year": year,
                "group": group,
                "text": text,
            })

            for sent in sentence_split(text):
                tokens = tokenize(sent)
                phrases = generate_ngrams(tokens)

                for phrase in phrases:
                    phrase_counter[phrase] += 1
                    phrase_papers[phrase].add(paper_id)
                    phrase_years[phrase].add(year)
                    phrase_groups[phrase].add(group)

                    if len(phrase_contexts[phrase]) < 3:
                        phrase_contexts[phrase].append(sent)

    phrase_rows = []

    for phrase, count in phrase_counter.items():
        paper_count = len(phrase_papers[phrase])

        if paper_count < MIN_GLOBAL_SUPPORT:
            continue

        phrase_rows.append({
            "phrase": phrase,
            "phrase_type": infer_phrase_type(phrase),
            "mention_count": count,
            "paper_count": paper_count,
            "groups": ";".join(sorted(phrase_groups[phrase])),
            "years": ";".join(map(str, sorted(phrase_years[phrase]))),
            "paper_ids": ";".join(sorted(phrase_papers[phrase])),
            "example_contexts": " || ".join(phrase_contexts[phrase]),
        })

    phrase_df = pd.DataFrame(phrase_rows)

    if phrase_df.empty:
        print("No n-gram phrases passed filtering.")
        return

    phrase_df = phrase_df.sort_values(
        ["paper_count", "mention_count"],
        ascending=False,
    )

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phrase_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Convert phrases into entity-style rows
    entity_rows = []

    valid_phrases = set(phrase_df["phrase"])

    for record in paper_texts:
        paper_id = record["paper_id"]
        year = record["year"]
        group = record["group"]
        text = record["text"].lower()

        for phrase in valid_phrases:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])"

            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                entity_rows.append({
                    "paper_id": paper_id,
                    "year": year,
                    "group": group,
                    "source": "title_abstract_ngram",
                    "entity_text": match.group(0),
                    "entity_normalized": phrase,
                    "entity_type": infer_phrase_type(phrase),
                    "relation_hint": "",
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start()-100): min(len(text), match.end()+100)],
                })

    entity_df = pd.DataFrame(entity_rows)

    entity_out = Path(OUTPUT_ENTITY_FILE)
    entity_df.to_csv(entity_out, index=False, encoding="utf-8-sig")

    print(f"Saved phrase table: {out_path}")
    print(f"Saved phrase entities: {entity_out}")
    print()
    print(f"Total retained phrases: {len(phrase_df)}")
    print(f"Total phrase entity mentions: {len(entity_df)}")
    print()
    print("Top phrases:")
    print(
        phrase_df[
            ["phrase", "phrase_type", "mention_count", "paper_count"]
        ].head(40).to_string(index=False)
    )


if __name__ == "__main__":
    main()
