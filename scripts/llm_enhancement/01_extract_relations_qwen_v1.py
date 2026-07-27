import json
import re
import time
import sys
from pathlib import Path

import pandas as pd
import requests

# ======================================================
# PATHS
# ======================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

# Allow imports from project root
sys.path.append(str(BASE))

from scripts.llm_enhancement.stx_ontology_utils import (
    normalize_entity,
    infer_entity_type,
    normalize_relation,
)

INPUT_FILE = BASE / "FINAL_WORKSPACE/processed/stx_enriched_master_corpus_FINAL.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/llm_outputs/v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RELATIONS = OUT_DIR / "qwen_relations_test10_corrected.csv"
OUT_EDGES = OUT_DIR / "qwen_typed_edges_test10_corrected.csv"
OUT_NODES = OUT_DIR / "qwen_nodes_test10_corrected.csv"
OUT_JSONL = OUT_DIR / "qwen_raw_test10_corrected.jsonl"
OUT_FAILED = OUT_DIR / "qwen_failed_test10_corrected.csv"

# ======================================================
# SETTINGS
# ======================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

N_RECORDS = 10
MIN_CONFIDENCE = 0.50
SLEEP_SECONDS = 0.3


# ======================================================
# HELPERS
# ======================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def build_prompt(title, abstract):
    return f"""
You are a marine toxin information extraction system.

Extract only explicitly supported saxitoxin-related biological relationships from the title and abstract.

Return valid JSON only. Do not include markdown, explanations, comments, or extra text.

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
2. Do not use general biological knowledge outside the text.
3. Do not invent evidence sentences.
4. Use confidence between 0 and 1.
5. If no saxitoxin-related relationship exists, return {{"relations": []}}.
6. Use normalized names where possible:
   - STX = saxitoxin
   - PST/PSTs = paralytic shellfish toxins
   - GTX = gonyautoxin
   - neoSTX = neosaxitoxin
   - sxtA1/sxtA4 = sxtA

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

    response = requests.post(OLLAMA_URL, json=payload, timeout=240)
    response.raise_for_status()
    return response.json()["response"]


def parse_json_response(text):
    text = str(text).strip()

    # Remove Qwen thinking block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {text[:500]}")
        return json.loads(match.group(0))


def validate_and_correct_relation(rel):
    raw_source = clean_text(rel.get("source_entity", ""))
    raw_target = clean_text(rel.get("target_entity", ""))

    source = normalize_entity(raw_source)
    target = normalize_entity(raw_target)

    if not source or not target or source == target:
        return None

    confidence = safe_float(rel.get("confidence", 0.0))
    if confidence < MIN_CONFIDENCE:
        return None

    source_type = infer_entity_type(source, rel.get("source_type", "OTHER"))
    target_type = infer_entity_type(target, rel.get("target_type", "OTHER"))
    relation_type = normalize_relation(rel.get("relation_type", "associated_with"))

    evidence = clean_text(rel.get("evidence_sentence", ""))

    if not evidence:
        return None

    return {
        "source": source,
        "source_label": raw_source,
        "source_type": source_type,
        "relation_type": relation_type,
        "target": target,
        "target_label": raw_target,
        "target_type": target_type,
        "evidence_sentence": evidence,
        "confidence": confidence,
    }


# ======================================================
# MAIN
# ======================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["title", "abstract", "year", "document_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Available columns: {df.columns.tolist()}")

    # For first test, use first 10 records
    df = df.head(N_RECORDS).copy()

    relation_rows = []
    failed_rows = []

    with open(OUT_JSONL, "w", encoding="utf-8") as jsonl:
        for i, row in df.iterrows():
            paper_id = clean_text(row.get("document_id", i))
            year = clean_text(row.get("year", ""))
            title = clean_text(row.get("title", ""))
            abstract = clean_text(row.get("abstract", ""))

            print(f"\nProcessing {len(relation_rows) + 1} | paper {i + 1}/{len(df)}: {paper_id}")

            if not title and not abstract:
                continue

            prompt = build_prompt(title, abstract)

            try:
                raw_response = call_qwen(prompt)
                parsed = parse_json_response(raw_response)
                raw_relations = parsed.get("relations", [])

                corrected_relations = []

                for rel in raw_relations:
                    corrected = validate_and_correct_relation(rel)
                    if corrected is None:
                        continue

                    corrected["paper_id"] = paper_id
                    corrected["year"] = year
                    corrected["title"] = title

                    corrected_relations.append(corrected)
                    relation_rows.append(corrected)

                jsonl.write(
                    json.dumps(
                        {
                            "paper_id": paper_id,
                            "year": year,
                            "title": title,
                            "raw_response": raw_response,
                            "corrected_relations": corrected_relations,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                print(f"Raw relations: {len(raw_relations)} | retained: {len(corrected_relations)}")

            except Exception as e:
                print(f"FAILED: {e}")
                failed_rows.append(
                    {
                        "paper_id": paper_id,
                        "year": year,
                        "title": title,
                        "error": str(e),
                    }
                )

            time.sleep(SLEEP_SECONDS)

    rel_df = pd.DataFrame(relation_rows)

    if len(rel_df) > 0:
        rel_df = rel_df.drop_duplicates(
            subset=["paper_id", "source", "relation_type", "target"]
        )

        rel_df.to_csv(OUT_RELATIONS, index=False)

        # KG-ready typed edges
        edge_df = (
            rel_df.groupby(
                ["source", "source_type", "relation_type", "target", "target_type"],
                as_index=False,
            )
            .agg(
                evidence_count=("paper_id", "nunique"),
                mean_confidence=("confidence", "mean"),
                evidence_examples=(
                    "evidence_sentence",
                    lambda x: " || ".join(list(x.dropna().astype(str).head(3))),
                ),
            )
        )

        edge_df.to_csv(OUT_EDGES, index=False)

        # KG-ready nodes
        source_nodes = rel_df[["source", "source_type"]].rename(
            columns={"source": "node", "source_type": "entity_type"}
        )
        target_nodes = rel_df[["target", "target_type"]].rename(
            columns={"target": "node", "target_type": "entity_type"}
        )

        nodes = (
            pd.concat([source_nodes, target_nodes], ignore_index=True)
            .drop_duplicates()
            .sort_values("node")
        )

        nodes.to_csv(OUT_NODES, index=False)

    else:
        pd.DataFrame().to_csv(OUT_RELATIONS, index=False)
        pd.DataFrame().to_csv(OUT_EDGES, index=False)
        pd.DataFrame().to_csv(OUT_NODES, index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(OUT_FAILED, index=False)

    print("\nDone.")
    print(f"Relations saved: {OUT_RELATIONS}")
    print(f"Typed edges saved: {OUT_EDGES}")
    print(f"Nodes saved: {OUT_NODES}")
    print(f"Raw JSONL saved: {OUT_JSONL}")

    if failed_rows:
        print(f"Failed records saved: {OUT_FAILED}")

    print(f"Total retained relations: {len(rel_df)}")


if __name__ == "__main__":
    main()
