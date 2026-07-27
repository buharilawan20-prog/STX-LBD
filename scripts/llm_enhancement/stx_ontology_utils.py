import re
import pandas as pd


DINO_TAXA = [
    "alexandrium", "gymnodinium", "pyrodinium", "gonyaulax",
    "dinophysis", "prorocentrum", "karenia", "coolia",
    "ostreopsis", "centrodinium", "protoceratium",
    "gambierdiscus", "amphidinium", "ceratium", "peridinium",
    "palatinus"
]

CYANO_TAXA = [
    "anabaena", "aphanizomenon", "cylindrospermopsis",
    "raphidiopsis", "dolichospermum", "lyngbya",
    "nostoc", "planktothrix", "microcystis",
    "cyanobacteria", "cyanobacterium", "cyanobacterial"
]

SXT_GENES = {
    "sxta": "sxtA",
    "sxta1": "sxtA",
    "sxta2": "sxtA",
    "sxta3": "sxtA",
    "sxta4": "sxtA",
    "sxtb": "sxtB",
    "sxtd": "sxtD",
    "sxtg": "sxtG",
    "sxth": "sxtH/T",
    "sxtt": "sxtH/T",
    "sxth/t": "sxtH/T",
    "sxti": "sxtI",
    "sxts": "sxtS",
    "sxtu": "sxtU",
    "sxt_genes": "sxt_genes",
    "sxtgenes": "sxt_genes",
}

TOXINS = {
    "stx": "saxitoxin",
    "saxitoxins": "saxitoxin",
    "saxitoxin": "saxitoxin",
    "pst": "paralytic_shellfish_toxins",
    "psts": "paralytic_shellfish_toxins",
    "paralytic_shellfish_toxin": "paralytic_shellfish_toxins",
    "paralytic_shellfish_toxins": "paralytic_shellfish_toxins",
    "gtx": "gonyautoxin",
    "gonyautoxins": "gonyautoxin",
    "gonyautoxin": "gonyautoxin",
    "neostx": "neosaxitoxin",
    "neo_stx": "neosaxitoxin",
    "neosaxitoxin": "neosaxitoxin",
}

ENV_FACTORS = [
    "temperature", "warming", "salinity", "light", "irradiance",
    "nitrate", "nitrogen", "phosphate", "phosphorus",
    "nutrient", "nutrients", "climate", "environmental_stress"
]

BIO_PROCESSES = [
    "biosynthesis", "saxitoxin_biosynthesis", "toxin_biosynthesis",
    "toxin_production", "gene_expression", "regulation",
    "transcriptomics", "transcriptome", "evolution",
    "horizontal_gene_transfer", "resting_cyst", "bloom",
    "harmful_algal_bloom"
]


def clean_entity(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-zA-Z0-9_/]+", "", x)
    return x


def normalize_entity(x):
    x = clean_entity(x)

    if x in SXT_GENES:
        return SXT_GENES[x]

    if x in TOXINS:
        return TOXINS[x]

    if x in ["hab", "habs"]:
        return "harmful_algal_bloom"

    return x


def infer_entity_type(entity, original_type="OTHER"):
    e = clean_entity(entity)

    if e in SXT_GENES or normalize_entity(e) in SXT_GENES.values():
        return "SXT_GENE"

    if e in TOXINS or normalize_entity(e) in TOXINS.values():
        return "TOXIN"

    if any(t in e for t in DINO_TAXA):
        return "DINO_TAXON"

    if any(t in e for t in CYANO_TAXA):
        return "CYANO_TAXON"

    if e in ENV_FACTORS:
        return "ENV_FACTOR"

    if e in BIO_PROCESSES:
        return "BIOLOGICAL_PROCESS"

    t = str(original_type).upper().replace(" ", "_")
    allowed = {
        "DINO_TAXON", "CYANO_TAXON", "SXT_GENE", "TOXIN",
        "ENV_FACTOR", "BIOLOGICAL_PROCESS", "DETECTION_METHOD", "OTHER"
    }

    return t if t in allowed else "OTHER"


def normalize_relation(x):
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")

    allowed = {
        "produces", "expresses", "regulates", "upregulates",
        "downregulates", "associated_with", "involved_in",
        "inhibits", "promotes", "detected_by", "affects",
        "contains_gene"
    }

    return x if x in allowed else "associated_with"
