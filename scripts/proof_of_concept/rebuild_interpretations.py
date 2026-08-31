
#!/usr/bin/env python3
"""Regenerate biologically directional interpretations in searchable_hypotheses.csv."""

from pathlib import Path
import shutil

import pandas as pd

from biological_interpretation import generate_biological_interpretation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE = (
    PROJECT_ROOT
    / "FINAL_WORKSPACE"
    / "proof_of_concept"
    / "searchable_hypotheses.csv"
)


def main() -> None:
    if not DATABASE.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE}")

    backup = DATABASE.with_name("searchable_hypotheses_before_interpretation_update.csv")
    if not backup.exists():
        shutil.copy2(DATABASE, backup)
        print(f"Backup created: {backup}")

    df = pd.read_csv(DATABASE)

    required = [
        "Query_Entity",
        "Query_Entity_Type",
        "Predicted_Entity",
        "Predicted_Entity_Type",
        "Hypothesis_Class",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

    df["Interpretation"] = df.apply(
        lambda row: generate_biological_interpretation(
            entity_a=row["Query_Entity"],
            type_a=row["Query_Entity_Type"],
            entity_b=row["Predicted_Entity"],
            type_b=row["Predicted_Entity_Type"],
            hypothesis_class=row.get("Hypothesis_Class", ""),
            bridge_nodes=row.get("Bridge_Nodes", ""),
        ),
        axis=1,
    )

    df.to_csv(DATABASE, index=False, encoding="utf-8-sig")
    print(f"Updated {len(df):,} interpretations: {DATABASE}")


if __name__ == "__main__":
    main()
