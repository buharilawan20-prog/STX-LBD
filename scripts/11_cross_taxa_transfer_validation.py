from pathlib import Path
import pandas as pd


# =====================================================
# INPUT FILES
# =====================================================
CYANO_FILE = "data/graphs_cyano/cyano_semantic_edges_normalized_collapsed.csv"

DINO_ALL_FILE = "data/graphs_dino/dino_semantic_edges_normalized_collapsed.csv"

DINO_PRE2016_FILE = (
    "results/dino_temporal_normalized_v2/dino_train_pre2016_edges.csv"
)

DINO_POST2015_FILE = (
    "results/dino_temporal_normalized_v2/dino_test_post2015_edges.csv"
)


# =====================================================
# OUTPUT
# =====================================================
OUTDIR = Path("results/cross_taxa_transfer")
OUTDIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# HELPERS
# =====================================================
def edge_key(a, b):
    return tuple(sorted([
        str(a).strip().lower(),
        str(b).strip().lower()
    ]))


def prepare(df):
    df = df.copy()

    df["Edge_Key"] = df.apply(
        lambda r: edge_key(r["Source"], r["Target"]),
        axis=1
    )

    return df


def run_transfer(
    train_df,
    test_df,
    label,
    prefix
):
    train_df = prepare(train_df)
    test_df = prepare(test_df)

    test_keys = set(test_df["Edge_Key"])

    train_df["Transfer_Match"] = train_df["Edge_Key"].apply(
        lambda x: 1 if x in test_keys else 0
    )

    total = len(train_df)
    matched = int(train_df["Transfer_Match"].sum())
    rate = matched / total if total else 0

    summary = pd.DataFrame([{
        "Analysis": label,
        "Train_Edges": total,
        "Matched_Edges": matched,
        "Transfer_Rate": rate
    }])

    summary.to_csv(
        OUTDIR / f"{prefix}_summary.csv",
        index=False
    )

    conserved = (
        train_df[train_df["Transfer_Match"] == 1]
        .sort_values(
            ["Weight", "Paper_Count"],
            ascending=False
        )
    )

    divergent = (
        train_df[train_df["Transfer_Match"] == 0]
        .sort_values(
            ["Weight", "Paper_Count"],
            ascending=False
        )
    )

    conserved.to_csv(
        OUTDIR / f"{prefix}_conserved.csv",
        index=False
    )

    divergent.to_csv(
        OUTDIR / f"{prefix}_divergent.csv",
        index=False
    )

    print("\n================================================")
    print(label)
    print("================================================")

    print(summary.to_string(index=False))

    print("\nTop conserved:")
    print(
        conserved[
            [
                "Source",
                "Target",
                "Weight",
                "Paper_Count"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nTop divergent:")
    print(
        divergent[
            [
                "Source",
                "Target",
                "Weight",
                "Paper_Count"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return summary


# =====================================================
# MAIN
# =====================================================
def main():

    print("\nLoading datasets...")

    cyano = pd.read_csv(CYANO_FILE, low_memory=False)

    dino_all = pd.read_csv(DINO_ALL_FILE, low_memory=False)

    dino_pre = pd.read_csv(
        DINO_PRE2016_FILE,
        low_memory=False
    )

    dino_post = pd.read_csv(
        DINO_POST2015_FILE,
        low_memory=False
    )

    all_summaries = []

    # =====================================================
    # 1. Cyano → ALL Dino
    # =====================================================
    s1 = run_transfer(
        cyano,
        dino_all,
        "Cyano → ALL Dino",
        "cyano_to_all_dino"
    )

    all_summaries.append(s1)

    # =====================================================
    # 2. Cyano → POST-2015 Dino
    # =====================================================
    s2 = run_transfer(
        cyano,
        dino_post,
        "Cyano → POST-2015 Dino",
        "cyano_to_post2015_dino"
    )

    all_summaries.append(s2)

    # =====================================================
    # 3. Cyano + PRE2016 Dino → POST2015 Dino
    # =====================================================
    combined_train = pd.concat(
        [cyano, dino_pre],
        ignore_index=True
    )

    combined_train = (
        combined_train
        .drop_duplicates(
            subset=["Source", "Target"]
        )
    )

    s3 = run_transfer(
        combined_train,
        dino_post,
        "Cyano + PRE2016 Dino → POST2015 Dino",
        "combined_to_post2015_dino"
    )

    all_summaries.append(s3)

    # =====================================================
    # SAVE OVERALL SUMMARY
    # =====================================================
    summary_df = pd.concat(
        all_summaries,
        ignore_index=True
    )

    summary_df.to_csv(
        OUTDIR / "overall_cross_taxa_summary.csv",
        index=False
    )

    print("\n================================================")
    print("OVERALL CROSS-TAXA SUMMARY")
    print("================================================")

    print(summary_df.to_string(index=False))

    print("\nSaved outputs in:")
    print(OUTDIR)


if __name__ == "__main__":
    main()
