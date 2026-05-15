from pathlib import Path
import pandas as pd


FILES = [
    "data/graphs_cyano/cyano_semantic_edges_normalized_filtered.csv",
    "data/graphs_dino/dino_semantic_edges_normalized_filtered.csv",
]


def normalize_key(a, b):
    return tuple(sorted([
        str(a).strip().lower(),
        str(b).strip().lower()
    ]))


def collapse_edges(df):

    df["collapsed_key"] = df.apply(
        lambda r: normalize_key(r["Source"], r["Target"]),
        axis=1
    )

    collapsed = (
        df.groupby("collapsed_key")
        .agg({
            "Source": "first",
            "Target": "first",
            "Relation": lambda x: ";".join(sorted(set(x))),
            "Source_Type": lambda x: ";".join(sorted(set(x))),
            "Target_Type": lambda x: ";".join(sorted(set(x))),
            "Weight": "sum",
            "Paper_Count": "sum",
            "Paper_IDs": lambda x: ";".join(sorted(set(";".join(x.astype(str)).split(";")))),
            "Years": lambda x: ";".join(sorted(set(";".join(x.astype(str)).split(";")))),
        })
        .reset_index(drop=True)
    )

    collapsed = collapsed.sort_values(
        ["Weight", "Paper_Count"],
        ascending=False
    )

    return collapsed


def main():

    for infile in FILES:

        path = Path(infile)

        print(f"\nProcessing: {path}")

        df = pd.read_csv(path, low_memory=False)

        collapsed = collapse_edges(df)

        outfile = path.with_name(
            path.stem.replace("_filtered", "_collapsed")
            + ".csv"
        )

        collapsed.to_csv(outfile, index=False)

        print(f"Saved: {outfile}")
        print(f"Original edges: {len(df)}")
        print(f"Collapsed edges: {len(collapsed)}")

        print("\nTop edges:")
        print(
            collapsed[
                ["Source", "Target", "Weight", "Paper_Count"]
            ]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
