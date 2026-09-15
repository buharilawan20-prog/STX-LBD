from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "proof_of_concept"
    / "searchable_hypotheses_v2.csv"
)


class STXLBD:

    METHOD_COLUMNS = {
        "Adamic–Adar": {
            "score": "AdamicAdar_Score",
            "rank": "AdamicAdar_Rank",
        },
        "Jaccard": {
            "score": "Jaccard_Score",
            "rank": "Jaccard_Rank",
        },
        "Node2Vec": {
            "score": "Node2Vec_Score",
            "rank": "Node2Vec_Rank",
        },
    }

    def __init__(self):

        if not DATABASE.exists():
            raise FileNotFoundError(
                f"STX-LBD database not found: {DATABASE}"
            )

        self.df = pd.read_csv(DATABASE)

        self.df["_source_norm"] = (
            self.df["Source"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.df["_target_norm"] = (
            self.df["Target"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    def search(
        self,
        entity,
        top_n=10,
        method="Adamic–Adar",
        temporal_filter="All hypotheses",
    ):

        entity_norm = (
            str(entity)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if method not in self.METHOD_COLUMNS:
            method = "Adamic–Adar"

        score_col = self.METHOD_COLUMNS[method]["score"]
        rank_col = self.METHOD_COLUMNS[method]["rank"]

        source_match = (
            self.df["_source_norm"]
            == entity_norm
        )

        target_match = (
            self.df["_target_norm"]
            == entity_norm
        )

        results = self.df[
            source_match | target_match
        ].copy()

        if results.empty:
            return results

        # ----------------------------------------------------
        # Present the opposite endpoint as the related entity.
        # ----------------------------------------------------

        def related_entity(row):

            if row["_source_norm"] == entity_norm:
                return row["Target"]

            return row["Source"]

        def related_type(row):

            if row["_source_norm"] == entity_norm:
                return row["Target_Type"]

            return row["Source_Type"]

        def query_type(row):

            if row["_source_norm"] == entity_norm:
                return row["Source_Type"]

            return row["Target_Type"]

        results["Query_Entity"] = entity_norm

        results["Query_Entity_Type"] = results.apply(
            query_type,
            axis=1,
        )

        results["Related_Entity"] = results.apply(
            related_entity,
            axis=1,
        )

        results["Related_Entity_Type"] = results.apply(
            related_type,
            axis=1,
        )

        # ----------------------------------------------------
        # Temporal filter
        # ----------------------------------------------------

        if temporal_filter != "All hypotheses":

            results = results[
                results["Temporal_Outcome"]
                == temporal_filter
            ]

        # ----------------------------------------------------
        # Selected graph method
        # ----------------------------------------------------

        results["Graph_Method"] = method

        results["Graph_Score"] = pd.to_numeric(
            results[score_col],
            errors="coerce",
        )

        results["Global_Graph_Rank"] = pd.to_numeric(
            results[rank_col],
            errors="coerce",
        )

        # Rank relative to the selected query results.
        results = results.sort_values(
            [
                "Graph_Score",
                "Global_Graph_Rank",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        ).copy()

        results["Rank_For_Query"] = range(
            1,
            len(results) + 1,
        )

        return results.head(top_n)

    def available_entities(self):

        entities = pd.concat(
            [
                self.df["Source"],
                self.df["Target"],
            ],
            ignore_index=True,
        )

        return sorted(
            entities
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
