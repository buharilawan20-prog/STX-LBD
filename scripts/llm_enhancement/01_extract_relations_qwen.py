import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT_FILES = [
    BASE / "FINAL_WORKSPACE/processed/stx_enriched_master_corpus_FINAL.csv",
]

OUT_DIR = BASE / "FINAL_WORKSPACE/llm_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RELATIONS = OUT_DIR / "qwen_test_relations_10.csv"
OUT_NODES = OUT_DIR / "qwen_test_nodes_10.csv"
OUT_EDGES = OUT_DIR / "qwen_test_typed_edges_10.csv"
OUT_JSONL = OUT_DIR / "qwen_test_raw_10.jsonl"
OUT_FAILED = OUT_DIR / "qwen_test_failed_10.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

N_RECORDS = 10
MIN_CONFIDENCE = 0.50


ENTITY_TYPES = {
    "DINO_TAXON",
    "CYANO_TAXON",
    "SXT_GENE",
    "TOXIN",
    "ENV_FACTOR",
    "BIOLOGICAL_PROCESS",
    "DETECTION_METHOD",
    "OTHER",
}

RELATION_TYPES = {
    "produces",
    "expresses",
    "regulates",
    "upregulates",
    "downregulates",
    "associated_with",
    "involved_in",
    "inhibits",
    "promotes",
    "detected_by",
    "affects",
    "contains_gene",
}


def find_input():
    for f in INPUT_FILES:
        if f.exists():
            return f
    raise FileNotFoundError("Corpus CSV not found. Edit INPUT_FILES in the script.")


def pick_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_entity(x):
    x = clean_text(x).lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-zA-Z0-9_/]+", "", x)

    mapping = {
        "pst": "paralytic_shellfish_toxins",
        "psts": "paralytic_shellfish_toxins",
        "paralytic_shellfish_toxin": "paralytic_shellfish_toxins",
        "paralytic_shellfish_toxins": "paralytic_shellfish_toxins",
        "stx": "saxitoxin",
        "gtx": "gonyautoxin",
        "neostx": "neosaxitoxin",
        "neo_stx": "neosaxitoxin",
        "sxta1": "sxtA",
        "sxta2": "sxtA",
        "sxta3": "sxtA",
        "sxta4": "sxtA",
        "sxta": "sxtA",
        "sxtg": "sxtG",
        "sxtd": "sxtD",
        "sxti": "sxtI",
        "sxtu": "sxtU",
        "sxth/t": "sxtH/T",
        "sxt_genes": "sxt_genes",
        "saxitoxin_biosynthesis": "saxitoxin_biosynthesis",
        "toxin_biosynthesis": "toxin_biosynthesis",
        "gene_expression": "gene_expression",
        "harmful_algal_bloom": "harmful_algal_bloom",
        "habs": "harmful_algal_blooms",
        "hab": "harmful_algal_bloom",
        "alexandrium_spp": "alexandrium",
        "alexandrium_sp": "alexandrium",
    }

    return mapping.get(x, x)


def normalize_relation(x):
    x = clean_text(x).lower().replace("-", "_").replace(" ", "_")
    if x not in RELATION_TYPES:
        return "associated_with"
    return x


def normalize_type(x):
    x = clean_text(x).upper().replace(" ", "_").replace("-", "_")
    if x not in ENTITY_TYPES:
        return "OTHER"
    return x


def build_prompt(title, abstract):
    return f"""
You are a marine toxin information extraction system.

Task:
Extract only explicitly supported saxitoxin-related biological relationships from the title and abstract.

Return valid JSON only. Do not include markdown, explanation, comments, or extra text.

Allowed entity types:
DINO_TAXON, CYANO_TAXON, SXT_GENE, TOXIN, ENV_FACTOR, BIOLOGICAL_PROCESS, DETECTION_METHOD, OTHER

Allowed relation types:
produces, expresses, regulates, upregulates, downregulates, associated_with, involved_in, inhibits, promotes, detected_by, affects, contains_gene

Required JSON schema:
{{
  "relations": [
    {{
      "source_entity": "string",
      "source_type": "DINO_TAXON|CYANO_TAXON|SXT_GENE|TOXIN|ENV_FACTOR|BIOLOGICAL_PROCESS|DETECTION_METHOD|OTHER",
      "relation_type": "produces|expresses|regulates|upregulates|downregulates|associated_with|involved_in|inhibits|promotes|detected_by|affects|contains_gene",
      "target_entity": "string",
      "target_type": "DINO_TAXON|CYANO_TAXON|SXT_GENE|TOXIN|ENV_FACTOR|BIOLOGICAL_PROCESS|DETECTION_METHOD|OTHER",
      "evidence_sentence": "string",
      "confidence": 0.0
    }}
  ]
}}

Rules:
1. Extract only relationships supported by the title or abstract.
2. Do not infer relationships from general background knowledge.
3. Do not invent evidence sentences.
4. Use confidence between 0 and 1.
5. Use normalized names where possible:
   - PST/PSTs = paralytic shellfish toxins
   - STX = saxitoxin
   - GTX = gonyautoxin
   - neoSTX = neosaxitoxin
   - sxtA1/sxtA4 = sxtA
6. If no saxitoxin-related relationship exists, return {{"relations": []}}.

Title:
{title}

Abstract:
{abstract}
""".strip()


def call_qwen(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()["response"]


def parse_json(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in response: {text[:500]}")
        return json.loads(m.group(0))


def validate_relation(rel):
    source = normalize_entity(rel.get("source_entity", ""))
    target = normalize_entity(rel.get("target_entity", ""))

    if not source or not target or source == target:
        return None

    try:
        confidence = float(rel.get("confidence", 0))
    except Exception:
        confidence = 0.0

    if confidence < MIN_CONFIDENCE:
        return None

    return {
        "source": source,
        "source_label": clean_text(rel.get("source_entity", "")),
        "source_type": normalize_type(rel.get("source_type", "")),
        "relation_type": normalize_relation(rel.get("relation_type", "")),
        "target": target,
        "target_label": clean_text(rel.get("target_entity", "")),
        "target_type": normalize_type(rel.get("target_type", "")),
        "evidence_sentence": clean_text(rel.get("evidence_sentence", "")),
        "confidence": confidence,
    }


def main():
    input_file = find_input()
    print("Reading:", input_file)

    df = pd.read_csv(input_file)

    title_col = pick_col(df, ["title", "Title"])
    abstract_col = pick_col(df, ["abstract", "Abstract", "abstract_text"])
    year_col = pick_col(df, ["year", "publication_year", "pub_year"])
    id_col = pick_col(df, ["paper_id", "pmid", "doi", "id"])

    if title_col is None or abstract_col is None:
        raise ValueError(f"Need title and abstract columns. Available: {df.columns.tolist()}")

    df = df.head(N_RECORDS).copy()

    relations_rows = []
    failed_rows = []

    with open(OUT_JSONL, "w", encoding="utf-8") as jf:
        for n, (_, row) in enumerate(df.iterrows(), start=1):
            paper_id = clean_text(row[id_col]) if id_col else str(n)
            year = clean_text(row[year_col]) if year_col else ""
            title = clean_text(row[title_col])
            abstract = clean_text(row[abstract_col])

            print(f"\nProcessing {n}/{len(df)}: {paper_id}")

            if not title and not abstract:
                continue

            prompt = build_prompt(title, abstract)

            try:
                raw = call_qwen(prompt)
                parsed = parse_json(raw)
                raw_relations = parsed.get("relations", [])

                cleaned_relations = []
                for rel in raw_relations:
                    valid = validate_relation(rel)
                    if valid:
                        valid["paper_id"] = paper_id
                        valid["year"] = year
                        valid["title"] = title
                        cleaned_relations.append(valid)
                        relations_rows.append(valid)

                jf.write(json.dumps({
                    "paper_id": paper_id,
                    "year": year,
                    "title": title,
                    "raw_response": raw,
                    "relations": cleaned_relations,
                }, ensure_ascii=False) + "\n")

                print("Raw relations:", len(raw_relations), "| retained:", len(cleaned_relations))

            except Exception as e:
                print("FAILED:", e)
                failed_rows.append({
                    "paper_id": paper_id,
                    "year": year,
                    "title": title,
                    "error": str(e),
                })

            time.sleep(0.3)

    rel_df = pd.DataFrame(relations_rows)

    if len(rel_df) > 0:
        rel_df = rel_df.drop_duplicates(
            subset=["source", "relation_type", "target", "paper_id"]
        )

        rel_df.to_csv(OUT_RELATIONS, index=False)

        edge_df = (
            rel_df.groupby(
                ["source", "source_type", "relation_type", "target", "target_type"],
                as_index=False
            )
            .agg(
                evidence_count=("paper_id", "nunique"),
                mean_confidence=("confidence", "mean"),
                evidence_examples=("evidence_sentence", lambda x: " || ".join(list(x.dropna().astype(str).head(3))))
            )
        )

        edge_df.to_csv(OUT_EDGES, index=False)

        source_nodes = rel_df[["source", "source_type"]].rename(
            columns={"source": "node", "source_type": "entity_type"}
        )
        target_nodes = rel_df[["target", "target_type"]].rename(
            columns={"target": "node", "target_type": "entity_type"}
        )

        nodes = pd.concat([source_nodes, target_nodes], ignore_index=True)
        nodes = nodes.drop_duplicates().sort_values("node")
        nodes.to_csv(OUT_NODES, index=False)

    else:
        pd.DataFrame().to_csv(OUT_RELATIONS, index=False)
        pd.DataFrame().to_csv(OUT_EDGES, index=False)
        pd.DataFrame().to_csv(OUT_NODES, index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(OUT_FAILED, index=False)

    print("\nDone.")
    print("Relations:", len(rel_df))
    print("Saved relations:", OUT_RELATIONS)
    print("Saved typed edges:", OUT_EDGES)
    print("Saved nodes:", OUT_NODES)
    print("Saved raw JSONL:", OUT_JSONL)
    if failed_rows:
        print("Saved failed:", OUT_FAILED)


if __name__ == "__main__":
    main()
