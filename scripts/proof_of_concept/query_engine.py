
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "proof_of_concept"
    / "searchable_hypotheses.csv"
)

class STXLBD:
    def __init__(self):
        self.df = pd.read_csv(DATABASE)

    def search(self, entity, top_n=10, validated_only=False):
        entity = entity.strip().lower()

        results = self.df[
            self.df["Query_Entity_Normalized"] == entity
        ].copy()

        if validated_only:
            results = results[
                results["Validation_Status"] == "Validated"
            ]

        results = results.sort_values(
            "AI_Score",
            ascending=False,
        )

        return results.head(top_n)
