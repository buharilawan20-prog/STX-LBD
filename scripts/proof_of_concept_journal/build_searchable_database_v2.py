#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RANKING = (
    ROOT
    / "FINAL_WORKSPACE"
    / "unsupervised_revision"
    / "dino_pre2016_unsupervised_rankings.csv"
)

TEMPORAL = (
    ROOT
    / "FINAL_WORKSPACE"
    / "unsupervised_revision"
    / "temporal_support_all_candidates.csv"
)

OUTDIR = (
    ROOT
    / "FINAL_WORKSPACE"
    / "proof_of_concept"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT = (
    OUTDIR
    / "searchable_hypotheses_v2.csv"
)

# ============================================================
# LOAD
# ============================================================

rank = pd.read_csv(RANKING)
temporal = pd.read_csv(TEMPORAL)

print("\n==========================================")
print("STX-LBD EXPLORER v2 DATABASE")
print("==========================================")

print("\nRanking:", rank.shape)
print("Temporal:", temporal.shape)

print("\nRanking columns:")
print(rank.columns.tolist())

print("\nTemporal columns:")
print(temporal.columns.tolist())

# ============================================================
# NORMALIZED PAIR
# ============================================================

def norm(x):
    return str(x).strip().lower()


def pair_key(a, b):

    pair = sorted(
        [
            norm(a),
            norm(b)
        ]
    )

    return "|||".join(pair)


rank["Pair_Key"] = rank.apply(
    lambda r: pair_key(
        r["Source"],
        r["Target"]
    ),
    axis=1
)

# ============================================================
# FIND TEMPORAL SOURCE/TARGET
# ============================================================

source_candidates = [
    "Source",
    "source",
    "Query_Entity"
]

target_candidates = [
    "Target",
    "target",
    "Predicted_Entity"
]


def find_col(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    return None


ts = find_col(
    temporal,
    source_candidates
)

tt = find_col(
    temporal,
    target_candidates
)

if ts is None or tt is None:

    raise ValueError(
        "Could not identify source/target columns "
        "in temporal-support table."
    )


temporal["Pair_Key"] = temporal.apply(
    lambda r: pair_key(
        r[ts],
        r[tt]
    ),
    axis=1
)

# ============================================================
# TEMPORAL OUTCOME
# ============================================================

status_col = find_col(
    temporal,
    [
        "Temporal_Status",
        "temporal_status",
        "Validation_Status",
        "Temporal_Supported",
        "Supported"
    ]
)

if status_col is None:

    raise ValueError(
        "Could not identify temporal outcome column."
    )


def temporal_label(value):

    s = str(value).strip().lower()

    if s in {
        "1",
        "true",
        "supported",
        "validated",
        "subsequent_support",
        "subsequent_dino_support",
        "represented",
        "subsequently_represented"
    }:

        return "Subsequently represented"

    return "Not observed post-2015"


temporal["Temporal_Outcome"] = (
    temporal[status_col]
    .apply(temporal_label)
)

temporal_small = (
    temporal[
        [
            "Pair_Key",
            "Temporal_Outcome"
        ]
    ]
    .drop_duplicates(
        "Pair_Key"
    )
)

# ============================================================
# JOIN — OUTCOME ONLY
#
# Important:
# Temporal outcome is NOT used to rank candidates.
# ============================================================

rank = rank.merge(
    temporal_small,
    on="Pair_Key",
    how="left"
)

rank["Temporal_Outcome"] = (
    rank["Temporal_Outcome"]
    .fillna(
        "Not assessed"
    )
)

# ============================================================
# INTERPRETATION
# ============================================================

def interpretation(row):

    source = str(
        row["Source"]
    ).replace("_", " ")

    target = str(
        row["Target"]
    ).replace("_", " ")

    bridges = str(
        row.get(
            "Bridge_Nodes",
            ""
        )
    )

    bridge_list = [
        x.strip().replace("_", " ")
        for x in bridges.split(";")
        if x.strip()
    ][:5]

    text = (
        f"The relationship between {source} and "
        f"{target} was identified as a candidate "
        f"missing link in the pre-2016 "
        f"dinoflagellate knowledge graph."
    )

    if bridge_list:

        text += (
            " Historical graph connectivity includes "
            "bridge concepts such as "
            + ", ".join(bridge_list)
            + "."
        )

    text += (
        " This relationship should be interpreted as "
        "a literature-derived hypothesis rather than "
        "experimental evidence of a biological mechanism."
    )

    return text


rank["Interpretation"] = rank.apply(
    interpretation,
    axis=1
)

# ============================================================
# DEFAULT RANK
#
# AA is the primary/default Explorer ranking because it
# performed best in the leakage-free temporal benchmark.
# ============================================================

rank["Primary_Rank"] = (
    rank["AdamicAdar_Rank"]
)

rank["Primary_Score"] = (
    rank["AdamicAdar_Score"]
)

rank["Primary_Method"] = (
    "Adamic–Adar"
)

# ============================================================
# SEARCH TEXT
# ============================================================

rank["Search_Text"] = (
    rank[
        [
            "Source",
            "Target",
            "Source_Type",
            "Target_Type",
            "Hypothesis_Class",
            "Bridge_Nodes",
            "Bridge_Types"
        ]
    ]
    .fillna("")
    .astype(str)
    .agg(
        " ".join,
        axis=1
    )
    .str.lower()
)

# ============================================================
# SORT
# ============================================================

rank = rank.sort_values(
    [
        "AdamicAdar_Rank",
        "Jaccard_Rank",
        "Node2Vec_Rank"
    ]
).reset_index(
    drop=True
)

# ============================================================
# SAVE
# ============================================================

rank.to_csv(
    OUT,
    index=False
)

print("\n==========================================")
print("DATABASE SUMMARY")
print("==========================================")

print(
    "Hypotheses:",
    len(rank)
)

print(
    "\nTemporal outcomes:"
)

print(
    rank["Temporal_Outcome"]
    .value_counts(
        dropna=False
    )
)

print(
    "\nPrimary ranking:",
    rank["Primary_Method"]
    .iloc[0]
)

print(
    "\nTop 10:"
)

print(
    rank[
        [
            "Primary_Rank",
            "Source",
            "Target",
            "Hypothesis_Class",
            "Primary_Score",
            "Temporal_Outcome"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)

print("\nSaved:")
print(OUT)
