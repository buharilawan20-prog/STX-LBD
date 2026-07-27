#!/usr/bin/env python3
"""
Build the STX-LBD searchable hypothesis database.

Primary input
-------------
dino_pre2016_hypotheses_ai_ranked.csv

Optional enrichment inputs
--------------------------
top_100_node2vec_hypotheses_for_interpretation.csv
strict_temporal_validated_hypotheses.csv

Outputs
-------
searchable_hypotheses.csv
searchable_hypotheses_unique.csv
searchable_entities.csv
database_summary.csv

The directional searchable database contains two rows per hypothesis:

    Source -> Target
    Target -> Source

This allows users to search for either member of a predicted relationship.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AI_FILE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "ml"
    / "dino_pre2016_hypotheses_ai_ranked.csv"
)

NODE2VEC_FILE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "ml"
    / "top_100_node2vec_hypotheses_for_interpretation.csv"
)

VALIDATED_FILE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "ml"
    / "strict_temporal_validated_hypotheses.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "proof_of_concept"
)

SEARCHABLE_OUTPUT = OUTPUT_DIR / "searchable_hypotheses.csv"

UNIQUE_OUTPUT = OUTPUT_DIR / "searchable_hypotheses_unique.csv"

ENTITIES_OUTPUT = OUTPUT_DIR / "searchable_entities.csv"

SUMMARY_OUTPUT = OUTPUT_DIR / "database_summary.csv"


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def normalize_entity(value: object) -> str:
    """
    Normalize an entity label for matching and searching.

    Original capitalization is preserved in display columns.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().casefold()
    text = text.replace("_", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"[^\w\s/+.-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_column_name(value: object) -> str:
    """Normalize a column name for flexible matching."""
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> str | None:
    """
    Find the first matching column from a list of possible names.
    """
    lookup = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            "\nCould not find a required column.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def load_csv(
    path: Path,
    required: bool = False,
) -> pd.DataFrame:
    """Load a CSV with clear reporting."""
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required input file was not found:\n{path}"
            )

        print(f"Optional file not found; skipping:\n  {path}")
        return pd.DataFrame()

    dataframe = pd.read_csv(path, low_memory=False)

    print(
        f"Loaded {path.name}: "
        f"{len(dataframe):,} rows × {len(dataframe.columns):,} columns"
    )

    return dataframe


def clean_text_series(
    dataframe: pd.DataFrame,
    column: str | None,
    default: str = "",
) -> pd.Series:
    """Return a clean string Series."""
    if column is None:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype="object",
        )

    return (
        dataframe[column]
        .fillna(default)
        .astype(str)
        .str.strip()
    )


def clean_numeric_series(
    dataframe: pd.DataFrame,
    column: str | None,
    default: float = np.nan,
) -> pd.Series:
    """Return a numeric Series."""
    if column is None:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype="float64",
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def make_pair_key(
    source: pd.Series,
    target: pd.Series,
) -> pd.Series:
    """
    Create an order-independent key.

    temperature + sxtA and sxtA + temperature receive the same key.
    """
    source_normalized = source.map(normalize_entity)
    target_normalized = target.map(normalize_entity)

    keys = [
        "|||".join(sorted([source_value, target_value]))
        for source_value, target_value
        in zip(source_normalized, target_normalized)
    ]

    return pd.Series(keys, index=source.index)


def first_nonempty(series: pd.Series) -> object:
    """Return the first meaningful value in a grouped Series."""
    for value in series:
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return value

    return ""


def standardize_validation_value(value: object) -> str:
    """Convert different temporal-label formats to a common status."""
    normalized = normalize_entity(value)

    positive_values = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "positive",
        "future positive",
        "future-positive",
        "future_positive",
        "validated",
        "temporally validated",
        "matched",
        "observed",
        "present",
    }

    negative_values = {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "negative",
        "future negative",
        "future-negative",
        "future_negative",
        "unvalidated",
        "not validated",
        "not observed",
        "absent",
    }

    if normalized in positive_values:
        return "Validated"

    if normalized in negative_values:
        return "Unvalidated"

    if "positive" in normalized or "validated" == normalized:
        return "Validated"

    if "negative" in normalized or "unvalidated" in normalized:
        return "Unvalidated"

    return "Not assessed"


# ============================================================
# LOAD THE PRIMARY AI-RANKED FILE
# ============================================================

try:
    ai = load_csv(AI_FILE, required=True)
except FileNotFoundError as error:
    print(error)
    print("\nSearch for the actual file with:")
    print(
        'find FINAL_WORKSPACE -iname '
        '"*dino*pre2016*ai*ranked*.csv"'
    )
    sys.exit(1)


# ============================================================
# IDENTIFY PRIMARY COLUMNS
# ============================================================

source_col = find_column(
    ai,
    ["Source", "Entity_1", "Node_1"],
    required=True,
)

target_col = find_column(
    ai,
    ["Target", "Entity_2", "Node_2"],
    required=True,
)

source_type_col = find_column(
    ai,
    ["Source_Type", "Entity_1_Type", "Node_1_Type"],
)

target_type_col = find_column(
    ai,
    ["Target_Type", "Entity_2_Type", "Node_2_Type"],
)

hypothesis_class_col = find_column(
    ai,
    [
        "Hypothesis_Class",
        "Hypothesis_Type",
        "Hypothesis_Category",
        "Relationship_Type",
        "Category",
    ],
)

final_ai_score_col = find_column(
    ai,
    [
        "Final_AI_Rank_Score",
        "AI_Score",
        "Final_AI_Score",
        "Predicted_Probability",
    ],
)

ml_probability_col = find_column(
    ai,
    [
        "ML_Probability",
        "Predicted_Probability",
        "Prediction_Probability",
        "Probability",
    ],
)

node2vec_score_col = find_column(
    ai,
    [
        "Node2Vec_Integrated_Score",
        "Node2Vec_Score",
        "Embedding_Integrated_Score",
    ],
)

embedding_st_col = find_column(
    ai,
    [
        "Embedding_Source_Target_Similarity",
        "Source_Target_Embedding_Similarity",
    ],
)

embedding_bridge_mean_col = find_column(
    ai,
    [
        "Embedding_Bridge_Mean_Similarity",
        "Bridge_Mean_Embedding_Similarity",
    ],
)

embedding_bridge_max_col = find_column(
    ai,
    [
        "Embedding_Bridge_Max_Similarity",
        "Bridge_Max_Embedding_Similarity",
    ],
)

embedding_coverage_col = find_column(
    ai,
    ["Embedding_Coverage"],
)

structural_score_col = find_column(
    ai,
    [
        "Score",
        "Structural_Score",
        "Graph_Score",
    ],
)

bridge_score_col = find_column(
    ai,
    ["Bridge_Score"],
)

common_neighbors_col = find_column(
    ai,
    ["Common_Neighbors"],
)

distinct_bridge_types_col = find_column(
    ai,
    ["Distinct_Bridge_Types"],
)

adamic_adar_col = find_column(
    ai,
    ["Adamic_Adar"],
)

jaccard_col = find_column(
    ai,
    ["Jaccard"],
)

preferential_attachment_col = find_column(
    ai,
    ["Preferential_Attachment"],
)

degree_source_col = find_column(
    ai,
    ["Degree_Source"],
)

degree_target_col = find_column(
    ai,
    ["Degree_Target"],
)

bridge_nodes_col = find_column(
    ai,
    [
        "Bridge_Nodes",
        "Intermediate_Nodes",
    ],
)

bridge_types_col = find_column(
    ai,
    ["Bridge_Types"],
)

training_graph_col = find_column(
    ai,
    ["Training_Graph"],
)

candidate_status_col = find_column(
    ai,
    ["Candidate_Status"],
)

temporal_label_col = find_column(
    ai,
    [
        "Temporal_Label",
        "Temporal_Validated",
        "Validation_Status",
        "Future_Match",
    ],
)

interpretation_col = find_column(
    ai,
    [
        "Interpretation",
        "Biological_Interpretation",
        "Hypothesis_Text",
    ],
)


# ============================================================
# BUILD THE UNIQUE HYPOTHESIS TABLE
# ============================================================

database = pd.DataFrame(
    {
        "Source": clean_text_series(ai, source_col),
        "Source_Type": clean_text_series(
            ai,
            source_type_col,
            default="Unknown",
        ),
        "Target": clean_text_series(ai, target_col),
        "Target_Type": clean_text_series(
            ai,
            target_type_col,
            default="Unknown",
        ),
        "Hypothesis_Class": clean_text_series(
            ai,
            hypothesis_class_col,
            default="Unclassified",
        ),
        "Final_AI_Rank_Score": clean_numeric_series(
            ai,
            final_ai_score_col,
        ),
        "ML_Probability": clean_numeric_series(
            ai,
            ml_probability_col,
        ),
        "Node2Vec_Integrated_Score": clean_numeric_series(
            ai,
            node2vec_score_col,
        ),
        "Embedding_Source_Target_Similarity":
            clean_numeric_series(ai, embedding_st_col),
        "Embedding_Bridge_Mean_Similarity":
            clean_numeric_series(ai, embedding_bridge_mean_col),
        "Embedding_Bridge_Max_Similarity":
            clean_numeric_series(ai, embedding_bridge_max_col),
        "Embedding_Coverage":
            clean_numeric_series(ai, embedding_coverage_col),
        "Structural_Score":
            clean_numeric_series(ai, structural_score_col),
        "Bridge_Score":
            clean_numeric_series(ai, bridge_score_col),
        "Common_Neighbors":
            clean_numeric_series(ai, common_neighbors_col),
        "Distinct_Bridge_Types":
            clean_numeric_series(ai, distinct_bridge_types_col),
        "Adamic_Adar":
            clean_numeric_series(ai, adamic_adar_col),
        "Jaccard":
            clean_numeric_series(ai, jaccard_col),
        "Preferential_Attachment":
            clean_numeric_series(ai, preferential_attachment_col),
        "Degree_Source":
            clean_numeric_series(ai, degree_source_col),
        "Degree_Target":
            clean_numeric_series(ai, degree_target_col),
        "Bridge_Nodes":
            clean_text_series(ai, bridge_nodes_col),
        "Bridge_Types":
            clean_text_series(ai, bridge_types_col),
        "Training_Graph":
            clean_text_series(
                ai,
                training_graph_col,
                default="dino_pre2016",
            ),
        "Candidate_Status":
            clean_text_series(
                ai,
                candidate_status_col,
                default="Candidate",
            ),
        "Interpretation":
            clean_text_series(ai, interpretation_col),
    }
)

database["Source_Normalized"] = (
    database["Source"].map(normalize_entity)
)

database["Target_Normalized"] = (
    database["Target"].map(normalize_entity)
)

database["Pair_Key"] = make_pair_key(
    database["Source"],
    database["Target"],
)

# Remove invalid rows.
database = database[
    database["Source_Normalized"].ne("")
    & database["Target_Normalized"].ne("")
    & database["Source_Normalized"].ne(
        database["Target_Normalized"]
    )
].copy()


# ============================================================
# DETERMINE TEMPORAL VALIDATION STATUS
# ============================================================

if temporal_label_col is not None:
    database["Validation_Status"] = (
        ai.loc[database.index, temporal_label_col]
        .map(standardize_validation_value)
    )
else:
    database["Validation_Status"] = "Unvalidated"


# ============================================================
# ENRICH WITH THE STRICTLY VALIDATED FILE
# ============================================================

validated = load_csv(
    VALIDATED_FILE,
    required=False,
)

if not validated.empty:
    validated_source_col = find_column(
        validated,
        ["Source", "Entity_1", "Node_1"],
        required=True,
    )

    validated_target_col = find_column(
        validated,
        ["Target", "Entity_2", "Node_2"],
        required=True,
    )

    validated_status_col = find_column(
        validated,
        [
            "Temporal_Validated",
            "Temporal_Label",
            "Validation_Status",
            "Validated",
            "Future_Match",
        ],
    )

    validated_table = pd.DataFrame(
        {
            "Source": clean_text_series(
                validated,
                validated_source_col,
            ),
            "Target": clean_text_series(
                validated,
                validated_target_col,
            ),
        }
    )

    validated_table["Pair_Key"] = make_pair_key(
        validated_table["Source"],
        validated_table["Target"],
    )

    if validated_status_col is not None:
        validated_table["Validated_File_Status"] = (
            validated[validated_status_col]
            .map(standardize_validation_value)
        )
    else:
        # The file contains only the validated subset.
        validated_table["Validated_File_Status"] = "Validated"

    validated_table = (
        validated_table[
            ["Pair_Key", "Validated_File_Status"]
        ]
        .drop_duplicates("Pair_Key")
    )

    database = database.merge(
        validated_table,
        on="Pair_Key",
        how="left",
    )

    validated_mask = (
        database["Validated_File_Status"] == "Validated"
    )

    database.loc[
        validated_mask,
        "Validation_Status",
    ] = "Validated"

    database = database.drop(
        columns=["Validated_File_Status"]
    )


# ============================================================
# OPTIONAL NODE2VEC INTERPRETATION ENRICHMENT
# ============================================================

node2vec = load_csv(
    NODE2VEC_FILE,
    required=False,
)

if not node2vec.empty:
    n2v_source_col = find_column(
        node2vec,
        ["Source", "Entity_1", "Node_1"],
        required=True,
    )

    n2v_target_col = find_column(
        node2vec,
        ["Target", "Entity_2", "Node_2"],
        required=True,
    )

    n2v_interpretation_col = find_column(
        node2vec,
        [
            "Interpretation",
            "Biological_Interpretation",
            "Hypothesis_Text",
        ],
    )

    n2v_temporal_col = find_column(
        node2vec,
        [
            "Temporal_Label",
            "Temporal_Validated",
            "Validation_Status",
        ],
    )

    n2v_table = pd.DataFrame(
        {
            "Source": clean_text_series(
                node2vec,
                n2v_source_col,
            ),
            "Target": clean_text_series(
                node2vec,
                n2v_target_col,
            ),
        }
    )

    n2v_table["Pair_Key"] = make_pair_key(
        n2v_table["Source"],
        n2v_table["Target"],
    )

    if n2v_interpretation_col is not None:
        n2v_table["Node2Vec_Interpretation"] = (
            clean_text_series(
                node2vec,
                n2v_interpretation_col,
            )
        )
    else:
        n2v_table["Node2Vec_Interpretation"] = ""

    if n2v_temporal_col is not None:
        n2v_table["Node2Vec_Validation_Status"] = (
            node2vec[n2v_temporal_col]
            .map(standardize_validation_value)
        )
    else:
        n2v_table["Node2Vec_Validation_Status"] = (
            "Not assessed"
        )

    n2v_table = (
        n2v_table[
            [
                "Pair_Key",
                "Node2Vec_Interpretation",
                "Node2Vec_Validation_Status",
            ]
        ]
        .drop_duplicates("Pair_Key")
    )

    database = database.merge(
        n2v_table,
        on="Pair_Key",
        how="left",
    )

    missing_interpretation = (
        database["Interpretation"].eq("")
        & database["Node2Vec_Interpretation"]
        .fillna("")
        .ne("")
    )

    database.loc[
        missing_interpretation,
        "Interpretation",
    ] = database.loc[
        missing_interpretation,
        "Node2Vec_Interpretation",
    ]

    node2vec_validated = (
        database["Node2Vec_Validation_Status"]
        == "Validated"
    )

    database.loc[
        node2vec_validated,
        "Validation_Status",
    ] = "Validated"

    database = database.drop(
        columns=[
            "Node2Vec_Interpretation",
            "Node2Vec_Validation_Status",
        ]
    )


# ============================================================
# SELECT THE MAIN USER-FACING SCORE
# ============================================================

database["AI_Score"] = database[
    "Final_AI_Rank_Score"
]

missing_ai_score = database["AI_Score"].isna()

database.loc[
    missing_ai_score,
    "AI_Score",
] = database.loc[
    missing_ai_score,
    "ML_Probability",
]

still_missing = database["AI_Score"].isna()

database.loc[
    still_missing,
    "AI_Score",
] = database.loc[
    still_missing,
    "Node2Vec_Integrated_Score",
]


# ============================================================
# REMOVE DUPLICATE RELATIONSHIPS
# ============================================================

database = database.sort_values(
    by=[
        "AI_Score",
        "ML_Probability",
        "Node2Vec_Integrated_Score",
    ],
    ascending=False,
    na_position="last",
)

database = database.drop_duplicates(
    subset=["Pair_Key"],
    keep="first",
).reset_index(drop=True)

database["Global_AI_Rank"] = (
    database["AI_Score"]
    .rank(
        method="first",
        ascending=False,
        na_option="bottom",
    )
    .astype(int)
)


# ============================================================
# CREATE SEARCHABLE TEXT
# ============================================================

search_columns = [
    "Source",
    "Target",
    "Source_Type",
    "Target_Type",
    "Hypothesis_Class",
    "Bridge_Nodes",
    "Bridge_Types",
    "Interpretation",
    "Validation_Status",
]

database["Search_Text"] = (
    database[search_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .map(normalize_entity)
)


# ============================================================
# CREATE BIDIRECTIONAL SEARCH ROWS
# ============================================================

forward = database.copy()

forward["Query_Entity"] = forward["Source"]
forward["Query_Entity_Type"] = forward["Source_Type"]
forward["Predicted_Entity"] = forward["Target"]
forward["Predicted_Entity_Type"] = forward["Target_Type"]
forward["Query_Direction"] = "Source-to-Target"

reverse = database.copy()

reverse["Query_Entity"] = reverse["Target"]
reverse["Query_Entity_Type"] = reverse["Target_Type"]
reverse["Predicted_Entity"] = reverse["Source"]
reverse["Predicted_Entity_Type"] = reverse["Source_Type"]
reverse["Query_Direction"] = "Target-to-Source"

searchable = pd.concat(
    [forward, reverse],
    ignore_index=True,
)

searchable["Query_Entity_Normalized"] = (
    searchable["Query_Entity"].map(normalize_entity)
)

searchable["Predicted_Entity_Normalized"] = (
    searchable["Predicted_Entity"].map(normalize_entity)
)

searchable = searchable.sort_values(
    by=[
        "Query_Entity_Normalized",
        "AI_Score",
        "ML_Probability",
        "Node2Vec_Integrated_Score",
    ],
    ascending=[True, False, False, False],
    na_position="last",
).reset_index(drop=True)

searchable["Rank_For_Query"] = (
    searchable
    .groupby("Query_Entity_Normalized")
    .cumcount()
    + 1
)


# ============================================================
# CREATE ENTITY AUTOCOMPLETE TABLE
# ============================================================

source_entities = database[
    ["Source", "Source_Type"]
].rename(
    columns={
        "Source": "Entity",
        "Source_Type": "Entity_Type",
    }
)

target_entities = database[
    ["Target", "Target_Type"]
].rename(
    columns={
        "Target": "Entity",
        "Target_Type": "Entity_Type",
    }
)

entities = pd.concat(
    [source_entities, target_entities],
    ignore_index=True,
)

entities["Entity_Normalized"] = (
    entities["Entity"].map(normalize_entity)
)

entities = (
    entities[
        entities["Entity_Normalized"].ne("")
    ]
    .drop_duplicates("Entity_Normalized")
    .sort_values(
        ["Entity_Type", "Entity"],
        ascending=True,
    )
    .reset_index(drop=True)
)

relationship_counts = pd.concat(
    [
        database["Source_Normalized"],
        database["Target_Normalized"],
    ],
    ignore_index=True,
).value_counts()

entities["Relationship_Count"] = (
    entities["Entity_Normalized"]
    .map(relationship_counts)
    .fillna(0)
    .astype(int)
)


# ============================================================
# COLUMN ORDER
# ============================================================

unique_column_order = [
    "Global_AI_Rank",
    "Source",
    "Source_Type",
    "Target",
    "Target_Type",
    "Hypothesis_Class",
    "AI_Score",
    "Final_AI_Rank_Score",
    "ML_Probability",
    "Node2Vec_Integrated_Score",
    "Embedding_Source_Target_Similarity",
    "Embedding_Bridge_Mean_Similarity",
    "Embedding_Bridge_Max_Similarity",
    "Embedding_Coverage",
    "Structural_Score",
    "Bridge_Score",
    "Common_Neighbors",
    "Distinct_Bridge_Types",
    "Adamic_Adar",
    "Jaccard",
    "Preferential_Attachment",
    "Degree_Source",
    "Degree_Target",
    "Validation_Status",
    "Candidate_Status",
    "Bridge_Nodes",
    "Bridge_Types",
    "Interpretation",
    "Training_Graph",
    "Source_Normalized",
    "Target_Normalized",
    "Pair_Key",
    "Search_Text",
]

database = database[
    [
        column
        for column in unique_column_order
        if column in database.columns
    ]
]

searchable_column_order = [
    "Query_Entity",
    "Query_Entity_Type",
    "Predicted_Entity",
    "Predicted_Entity_Type",
    "Rank_For_Query",
    "Global_AI_Rank",
    "Hypothesis_Class",
    "AI_Score",
    "Final_AI_Rank_Score",
    "ML_Probability",
    "Node2Vec_Integrated_Score",
    "Validation_Status",
    "Candidate_Status",
    "Bridge_Nodes",
    "Bridge_Types",
    "Interpretation",
    "Common_Neighbors",
    "Distinct_Bridge_Types",
    "Structural_Score",
    "Bridge_Score",
    "Embedding_Source_Target_Similarity",
    "Embedding_Bridge_Mean_Similarity",
    "Embedding_Bridge_Max_Similarity",
    "Embedding_Coverage",
    "Adamic_Adar",
    "Jaccard",
    "Preferential_Attachment",
    "Degree_Source",
    "Degree_Target",
    "Source",
    "Source_Type",
    "Target",
    "Target_Type",
    "Training_Graph",
    "Query_Direction",
    "Query_Entity_Normalized",
    "Predicted_Entity_Normalized",
    "Pair_Key",
    "Search_Text",
]

searchable = searchable[
    [
        column
        for column in searchable_column_order
        if column in searchable.columns
    ]
]


# ============================================================
# BUILD SUMMARY
# ============================================================

validated_count = int(
    database["Validation_Status"]
    .eq("Validated")
    .sum()
)

unvalidated_count = int(
    database["Validation_Status"]
    .eq("Unvalidated")
    .sum()
)

not_assessed_count = int(
    database["Validation_Status"]
    .eq("Not assessed")
    .sum()
)

summary_rows = [
    {
        "Metric": "Unique hypotheses",
        "Value": len(database),
    },
    {
        "Metric": "Searchable directional rows",
        "Value": len(searchable),
    },
    {
        "Metric": "Unique searchable entities",
        "Value": entities["Entity_Normalized"].nunique(),
    },
    {
        "Metric": "Temporally validated hypotheses",
        "Value": validated_count,
    },
    {
        "Metric": "Unvalidated hypotheses",
        "Value": unvalidated_count,
    },
    {
        "Metric": "Not assessed hypotheses",
        "Value": not_assessed_count,
    },
    {
        "Metric": "Hypothesis classes",
        "Value": database["Hypothesis_Class"].nunique(),
    },
]

summary = pd.DataFrame(summary_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

database.to_csv(
    UNIQUE_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

searchable.to_csv(
    SEARCHABLE_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

entities.to_csv(
    ENTITIES_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# TERMINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("STX-LBD SEARCHABLE DATABASE CREATED SUCCESSFULLY")
print("=" * 72)

print(f"\nPrimary AI file:\n  {AI_FILE}")

print("\nOutputs:")
print(f"  Unique hypotheses:    {UNIQUE_OUTPUT}")
print(f"  Searchable database:  {SEARCHABLE_OUTPUT}")
print(f"  Searchable entities:  {ENTITIES_OUTPUT}")
print(f"  Database summary:     {SUMMARY_OUTPUT}")

print("\nDatabase statistics:")
print(f"  Unique hypotheses:           {len(database):,}")
print(f"  Directional search rows:     {len(searchable):,}")
print(f"  Unique searchable entities:  {len(entities):,}")
print(f"  Validated:                   {validated_count:,}")
print(f"  Unvalidated:                 {unvalidated_count:,}")
print(f"  Not assessed:                {not_assessed_count:,}")

print("\nTop 10 hypotheses:")

preview_columns = [
    "Global_AI_Rank",
    "Source",
    "Target",
    "Hypothesis_Class",
    "AI_Score",
    "Validation_Status",
]

print(
    database[preview_columns]
    .head(10)
    .to_string(index=False)
)

print("\nExample search command:")

print(
    """
python - <<'PY'
import pandas as pd

df = pd.read_csv(
    "FINAL_WORKSPACE/proof_of_concept/searchable_hypotheses.csv"
)

query = "sxtA".casefold()

results = df[
    df["Query_Entity_Normalized"].eq(query)
].head(10)

print(
    results[
        [
            "Rank_For_Query",
            "Predicted_Entity",
            "Hypothesis_Class",
            "AI_Score",
            "Validation_Status",
        ]
    ].to_string(index=False)
)
PY
"""
)
