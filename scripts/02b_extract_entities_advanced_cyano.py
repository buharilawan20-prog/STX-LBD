from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

INPUT_FILES = [
    "data/processed/cyano_all_clean.csv",
]

OUTPUT_FILE = "data/processed/entities_cyano_advanced.csv"

ENTITY_PATTERNS = {
    # =========================
    # SAXITOXIN GENES / DOMAINS
    # =========================
    "GENE": [
        r"\bsxtA\b", r"\bsxtA1\b", r"\bsxtA2\b", r"\bsxtA3\b", r"\bsxtA4\b",
        r"\bsxtB\b", r"\bsxtC\b", r"\bsxtD\b", r"\bsxtG\b", r"\bsxtH\b",
        r"\bsxtI\b", r"\bsxtN\b", r"\bsxtO\b", r"\bsxtP\b", r"\bsxtQ\b",
        r"\bsxtS\b", r"\bsxtT\b", r"\bsxtU\b", r"\bsxtV\b", r"\bsxtW\b",
        r"\bsxtX\b", r"\bsxtY\b", r"\bsxtZ\b",
        r"\bsxt\s*gene[s]?\b",
        r"\bsxt\s*homolog[s]?\b",
        r"\bcore\s+sxt\s+gene[s]?\b",
        r"\bsaxitoxin\s+biosynthesis\s+gene[s]?\b",
        r"\bSTX\s+biosynthesis\s+gene[s]?\b",
    ],

    "GENE_DOMAIN": [
        r"\bsxtA4\s+domain\b",
        r"\bA4\s+domain\b",
        r"\bSAM[-\s]?dependent\s+methyltransferase\b",
        r"\bclass\s*II\s+aminotransferase\b",
        r"\bamidinotransferase\b",
        r"\bshort[-\s]?chain\s+dehydrogenase\b",
        r"\bRossmann\s+fold\b",
        r"\btransferase\s+domain\b",
        r"\baminotransferase\s+domain\b",
        r"\bmethyltransferase\s+domain\b",
    ],

    # =========================
    # PKS / FAS / SECONDARY METABOLISM
    # =========================
    "BIOSYNTHETIC_SYSTEM": [
        r"\bPKS\b",
        r"\bpolyketide\s+synthase[s]?\b",
        r"\bFAS\b",
        r"\bfatty\s+acid\s+synthase[s]?\b",
        r"\bsecondary\s+metabolite[s]?\b",
        r"\bsecondary\s+metabolism\b",
        r"\bbiosynthetic\s+pathway[s]?\b",
        r"\bmetabolic\s+pathway[s]?\b",
    ],

    # =========================
    # SPECIES / TAXA
    # =========================
    "SPECIES": [
        r"\bGymnodinium\s+catenatum\b", r"\bG\.\s*catenatum\b",
        r"\bGymnodinium\s+impudicum\b", r"\bG\.\s*impudicum\b",
        r"\bGymnodinium\s+smaydae\b", r"\bG\.\s*smaydae\b",
        r"\bAlexandrium\s+catenella\b", r"\bA\.\s*catenella\b",
        r"\bAlexandrium\s+pacificum\b", r"\bA\.\s*pacificum\b",
        r"\bAlexandrium\s+tamarense\b", r"\bA\.\s*tamarense\b",
        r"\bAlexandrium\s+minutum\b", r"\bA\.\s*minutum\b",
        r"\bAlexandrium\s+fundyense\b", r"\bA\.\s*fundyense\b",
        r"\bAlexandrium\s+ostenfeldii\b", r"\bA\.\s*ostenfeldii\b",
        r"\bPyrodinium\s+bahamense\b", r"\bP\.\s*bahamense\b",
        r"\bCentrodinium\s+punctatum\b", r"\bC\.\s*punctatum\b",
        r"\bGonyaulax\s+spinifera\b", r"\bG\.\s*spinifera\b",
    ],

    "TAXON_GROUP": [
        r"\bdinoflagellate[s]?\b",
        r"\bmarine\s+dinoflagellate[s]?\b",
        r"\bfreshwater\s+dinoflagellate[s]?\b",
        r"\bcyanobacteria\b",
        r"\bcyanobacterial\b",
        r"\btoxic\s+cyanobacteria\b",
        r"\bGymnodinium\b",
        r"\bAlexandrium\b",
        r"\bPyrodinium\b",
    ],

    # =========================
    # TOXINS / TOXIN PHENOTYPES
    # =========================
    "TOXIN": [
        r"\bsaxitoxin\b",
        r"\bSTX\b",
        r"\bSTXs\b",
        r"\bparalytic\s+shellfish\s+toxin[s]?\b",
        r"\bPST\b",
        r"\bPSTs\b",
        r"\bparalytic\s+shellfish\s+poisoning\b",
        r"\bPSP\b",
        r"\bgonyautoxin\b",
        r"\bGTX\b",
        r"\bGTXs\b",
        r"\bneosaxitoxin\b",
        r"\bneoSTX\b",
        r"\bdecarbamoylsaxitoxin\b",
        r"\bdcSTX\b",
        r"\btoxin\s+profile[s]?\b",
        r"\btoxin\s+composition\b",
        r"\btoxin\s+content\b",
        r"\btoxin\s+production\b",
        r"\btoxin\s+accumulation\b",
    ],

    "TOXIN_PHENOTYPE": [
        r"\btoxigenic\b",
        r"\bnon[-\s]?toxigenic\b",
        r"\btoxic\b",
        r"\bnon[-\s]?toxic\b",
        r"\bSTX[-\s]?producing\b",
        r"\bnon[-\s]?STX[-\s]?producing\b",
        r"\bPST[-\s]?producing\b",
        r"\btoxin[-\s]?producing\b",
        r"\blacked\s+detectable\s+PSTs\b",
        r"\bno\s+detectable\s+PSTs\b",
        r"\bundetectable\s+toxin[s]?\b",
        r"\btoxin[s]?\s+not\s+detected\b",
    ],

    # =========================
    # GENE PRESENCE / ABSENCE / COPY NUMBER
    # =========================
    "GENE_PRESENCE_ABSENCE": [
        r"\bgene\s+presence\b",
        r"\bgene\s+absence\b",
        r"\bpresence\s+of\s+sxt\s+gene[s]?\b",
        r"\babsence\s+of\s+sxt\s+gene[s]?\b",
        r"\black(?:ed|s|ing)?\s+core\s+gene[s]?\b",
        r"\black(?:ed|s|ing)?\s+sxtA\b",
        r"\black(?:ed|s|ing)?\s+sxtG\b",
        r"\babsence\s+of\s+sxtB\b",
        r"\bgene\s+loss\b",
        r"\bgenus[-\s]?specific\s+gene\s+loss\b",
        r"\bcomplete\s+set\s+of\s+sxt\s+gene[s]?\b",
        r"\bretained\s+sxt\s+gene[s]?\b",
        r"\bsxt\s+repertoire\b",
        r"\bsxt\s+homolog[s]?\b",
        r"\bmultiple\s+homolog[s]?\b",
        r"\bgene\s+copy\s+number\b",
        r"\bgene\s+duplication\b",
        r"\bexpanded\s+gene\s+famil(?:y|ies)\b",
    ],

    # =========================
    # EXPRESSION / REGULATION
    # =========================
    "REGULATION_EXPRESSION": [
        r"\bgene\s+expression\b",
        r"\bexpression\s+level[s]?\b",
        r"\btranscriptional\s+response[s]?\b",
        r"\btranscriptional\s+regulation\b",
        r"\btranscriptomic\s+comparison\b",
        r"\bcomparative\s+transcriptomic[s]?\b",
        r"\btranscriptome\s+sequencing\b",
        r"\bup[-\s]?regulated\b",
        r"\bdown[-\s]?regulated\b",
        r"\bdifferential\s+expression\b",
        r"\bDEG[s]?\b",
        r"\bregulatory\s+factor[s]?\b",
        r"\bregulation\s+of\s+STX\b",
        r"\bpost[-\s]?transcriptional\s+regulation\b",
        r"\bepigenetic\s+regulation\b",
        r"\bexpression\s+pattern[s]?\b",
        r"\bhighest\s+expression\b",
        r"\blow\s+expression\b",
        r"\bundetectable\s+expression\b",
    ],

    # =========================
    # EVOLUTION / PHYLOGENY / HGT
    # =========================
    "EVOLUTIONARY_PROCESS": [
        r"\bevolution\b",
        r"\bevolutionary\s+relationship[s]?\b",
        r"\bevolutionary\s+divergence\b",
        r"\bphylogeny\b",
        r"\bphylogenetic\s+analysis\b",
        r"\bphylogenetic\s+tree\b",
        r"\bclade\b",
        r"\bgrouping\b",
        r"\bconserved\b",
        r"\bconservation\b",
        r"\bdivergence\b",
        r"\bdivergent\b",
        r"\bindependent\s+acquisition\b",
        r"\bindependent\s+evolution\b",
        r"\bhorizontal\s+gene\s+transfer\b",
        r"\bHGT\b",
        r"\bshared\s+ancestry\b",
        r"\bgene\s+transfer\b",
        r"\bortholog[s]?\b",
        r"\bparalog[s]?\b",
        r"\bhomolog[s]?\b",
        r"\bhomology\b",
        r"\bancestral\b",
    ],

    # =========================
    # ENVIRONMENT / ECOLOGY
    # =========================
    "ENV_FACTOR": [
        r"\btemperature\b",
        r"\bhigh\s+temperature\b",
        r"\blow\s+temperature\b",
        r"\bthermal\s+stress\b",
        r"\bsalinity\b",
        r"\bhigh\s+salinity\b",
        r"\blow\s+salinity\b",
        r"\blight\b",
        r"\birradiance\b",
        r"\bnitrogen\b",
        r"\bnitrate\b",
        r"\bammonium\b",
        r"\bphosphate\b",
        r"\bphosphorus\b",
        r"\bsilicate\b",
        r"\bnutrient\s+limitation\b",
        r"\bnitrogen\s+limitation\b",
        r"\bphosphorus\s+limitation\b",
        r"\bnutrient\s+stress\b",
        r"\bpH\b",
        r"\bCO2\b",
        r"\bcarbon\s+dioxide\b",
        r"\boxidative\s+stress\b",
        r"\bhypoxia\b",
        r"\bclimate\s+change\b",
        r"\bwarming\b",
        r"\benvironmental\s+condition[s]?\b",
        r"\benvironmental\s+factor[s]?\b",
    ],

    # =========================
    # BIOLOGICAL MECHANISM
    # =========================
    "BIOLOGICAL_MECHANISM": [
        r"\bbiosynthesis\b",
        r"\bSTX\s+biosynthesis\b",
        r"\bsaxitoxin\s+biosynthesis\b",
        r"\bPST\s+biosynthesis\b",
        r"\bbiosynthetic\s+mechanism[s]?\b",
        r"\bmetabolic\s+function[s]?\b",
        r"\benzyme[-\s]?catalyzed\s+reaction[s]?\b",
        r"\benzymatic\s+activity\b",
        r"\bamidinotransferase\s+activity\b",
        r"\btransferase\s+activity\b",
        r"\bmethyltransferase\s+activity\b",
        r"\baminotransferase\s+activity\b",
        r"\bdehydrogenase\s+activity\b",
        r"\bcell\s+cycle\b",
        r"\bG1\s+phase\b",
        r"\btoxin\s+biosynthesis\s+pathway\b",
        r"\bprecursor[s]?\b",
        r"\barginine\b",
        r"\bmethionine\b",
        r"\bS[-\s]?adenosylmethionine\b",
        r"\bacetate\b",
    ],
}


# Canonical map for common variants
CANONICAL_MAP = {
    "stx": "saxitoxin",
    "stxs": "saxitoxin",
    "pst": "paralytic shellfish toxins",
    "psts": "paralytic shellfish toxins",
    "paralytic shellfish toxin": "paralytic shellfish toxins",
    "psp": "paralytic shellfish poisoning",

    "g. catenatum": "Gymnodinium catenatum",
    "g. impudicum": "Gymnodinium impudicum",
    "g. smaydae": "Gymnodinium smaydae",
    "a. catenella": "Alexandrium catenella",
    "a. pacificum": "Alexandrium pacificum",
    "a. tamarense": "Alexandrium tamarense",
    "a. minutum": "Alexandrium minutum",
    "a. fundyense": "Alexandrium fundyense",
    "a. ostenfeldii": "Alexandrium ostenfeldii",
    "p. bahamense": "Pyrodinium bahamense",
    "c. punctatum": "Centrodinium punctatum",

    "sxta": "sxtA",
    "sxta1": "sxtA1",
    "sxta2": "sxtA2",
    "sxta3": "sxtA3",
    "sxta4": "sxtA4",
    "sxtb": "sxtB",
    "sxtg": "sxtG",
    "sxtu": "sxtU",
    "pks": "PKS",
    "polyketide synthase": "PKS",
    "fas": "FAS",
    "fatty acid synthase": "FAS",
}


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize(entity: str) -> str:
    entity_clean = re.sub(r"\s+", " ", entity.strip())
    key = entity_clean.lower().replace("-", " ")

    key = re.sub(r"\s+", " ", key)

    if key in CANONICAL_MAP:
        return CANONICAL_MAP[key]

    # Preserve gene capitalization
    gene_match = re.fullmatch(r"sxt([a-z])(\d?)", key)
    if gene_match:
        letter = gene_match.group(1).upper()
        number = gene_match.group(2)
        return f"sxt{letter}{number}"

    return entity_clean


def get_context(text: str, start: int, end: int, window: int = 100) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def extract_sentence_level_findings(text: str, paper_id: str, year: int, group: str, source: str):
    """
    Extract simple finding-like triples from sentences.
    These are not perfect, but they capture mechanistic statements better than co-occurrence.
    """
    rows = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    trigger_patterns = {
        "INCREASES": r"\b(increase[d|s]?|enhance[d|s]?|elevate[d|s]?|up[-\s]?regulated|higher)\b",
        "DECREASES": r"\b(decrease[d|s]?|reduce[d|s]?|lower|down[-\s]?regulated|suppressed)\b",
        "ASSOCIATED_WITH": r"\b(associated\s+with|correlated\s+with|linked\s+to|related\s+to)\b",
        "PRESENT_IN": r"\b(identified\s+in|detected\s+in|present\s+in|found\s+in|retained\s+in)\b",
        "ABSENT_IN": r"\b(absent\s+in|lacked\s+in|missing\s+in|not\s+detected\s+in|lacked)\b",
        "REGULATES": r"\b(regulate[d|s]?|controlled\s+by|influenced\s+by|modulated\s+by)\b",
        "INVOLVED_IN": r"\b(involved\s+in|participates\s+in|plays\s+a\s+role\s+in|contributes\s+to)\b",
        "EVOLUTIONARY_LINK": r"\b(clade|phylogenetic|evolutionary|divergence|conserved|independent\s+acquisition|HGT|horizontal\s+gene\s+transfer)\b",
    }

    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 20:
            continue

        for relation, pattern in trigger_patterns.items():
            if re.search(pattern, sent_clean, flags=re.IGNORECASE):
                rows.append({
                    "paper_id": paper_id,
                    "year": year,
                    "group": group,
                    "source": source,
                    "entity_text": sent_clean,
                    "entity_normalized": sent_clean,
                    "entity_type": "FINDING_SENTENCE",
                    "relation_hint": relation,
                    "start": 0,
                    "end": len(sent_clean),
                    "context": sent_clean,
                })

    return rows


def extract_entities_from_text(text: str, paper_id: str, year: int, group: str, source: str):
    rows = []

    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw = match.group(0)
                normalized = canonicalize(raw)

                rows.append({
                    "paper_id": paper_id,
                    "year": year,
                    "group": group,
                    "source": source,
                    "entity_text": raw,
                    "entity_normalized": normalized,
                    "entity_type": entity_type,
                    "relation_hint": "",
                    "start": match.start(),
                    "end": match.end(),
                    "context": get_context(text, match.start(), match.end()),
                })

    rows.extend(
        extract_sentence_level_findings(
            text=text,
            paper_id=paper_id,
            year=year,
            group=group,
            source=source,
        )
    )

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
        print("No entities found.")
        return

    ent_df = ent_df.drop_duplicates(
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
    ent_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved advanced entities: {output_path}")
    print(f"Total extracted records: {len(ent_df)}")
    print()
    print("Entity counts by type:")
    print(ent_df["entity_type"].value_counts())
    print()
    print("Top normalized entities:")
    print(ent_df["entity_normalized"].value_counts().head(40))
    print()
    print("Relation hints from finding sentences:")
    finding_df = ent_df[ent_df["entity_type"] == "FINDING_SENTENCE"]
    if len(finding_df) > 0:
        print(finding_df["relation_hint"].value_counts())
    else:
        print("No finding sentences extracted.")


if __name__ == "__main__":
    main()
