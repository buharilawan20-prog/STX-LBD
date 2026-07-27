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

    def search(
        self,
        entity,
        top_n=10,
        validated_only=False
    ):

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
            ascending=False
        )

        return results.head(top_n)

    def print_results(
        self,
        entity,
        top_n=10,
        validated_only=False
    ):

        results = self.search(
            entity,
            top_n,
            validated_only
        )

        if len(results) == 0:

            print("\nNo results found.\n")
            return

        print()

        print("=" * 70)

        print(f"Results for: {entity}")

        print("=" * 70)

        for i, row in results.iterrows():

            print()

            print(
                f"Rank: {row['Rank_For_Query']}"
            )

            print(
                f"Prediction: {row['Predicted_Entity']}"
            )

            print(
                f"AI Score: {row['AI_Score']:.3f}"
            )

            print(
                f"Validation: {row['Validation_Status']}"
            )

            print(
                f"Class: {row['Hypothesis_Class']}"
            )

            print(
                f"Interpretation:"
            )

            print(
                row["Interpretation"]
            )

            print("-" * 70)
