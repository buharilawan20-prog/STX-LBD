from pathlib import Path
import pandas as pd


CYANO_FILE = "data/graphs_cyano/cyano_semantic_edges_filtered.csv"
DINO_PRE_FILE = "../New/data/graphs_semantic/semantic_pre2016_edges_filtered.csv"

OUTPUT_FILE = "data/graphs_transfer/combined_cyano_dino_train_edges.csv"


def main():
    cyano = pd.read_csv(CYANO_FILE, encoding="utf-8-sig", low_memory=False)
    dino = pd.read_csv(DINO_PRE_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"Cyano edges: {len(cyano)}")
    print(f"Dino pre-2016 edges: {len(dino)}")

    combined = pd.concat([cyano, dino], ignore_index=True)

    # Normalize
    combined["Source"] = combined["Source"].astype(str).str.strip()
    combined["Target"] = combined["Target"].astype(str).str.strip()

    # Aggregate duplicates
    combined = (
        combined.groupby(
            ["Source", "Target", "Source_Type", "Target_Type", "Relation"],
            as_index=False
        )
        .agg(
            Weight=("Weight", "sum"),
            Paper_Count=("Paper_Count", "sum")
        )
    )

    Path("data/graphs_transfer").mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved combined graph: {OUTPUT_FILE}")
    print(f"Total combined edges: {len(combined)}")


if __name__ == "__main__":
    main()
