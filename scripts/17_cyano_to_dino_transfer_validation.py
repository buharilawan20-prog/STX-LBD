from pathlib import Path
import pandas as pd


CYANO_EDGE_FILE = "data/graphs_cyano/cyano_semantic_edges_filtered.csv"

DINO_POST2015_EDGE_FILE = "../New/data/graphs_semantic/semantic_post2015_edges_filtered.csv"
DINO_ALL_EDGE_FILE = "../New/data/graphs_semantic/semantic_all_edges_filtered.csv"

OUTPUT_DIR = "results/cyano_transfer"


def edge_key(a, b):
    return tuple(sorted([str(a).strip().lower(), str(b).strip().lower()]))


def clean_edges(df):
    df = df.copy()
    df["Source"] = df["Source"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()
    df["Edge_Key"] = df.apply(
        lambda r: edge_key(r["Source"], r["Target"]),
        axis=1
    )
    return df


def validate_transfer(cyano_df, dino_df):
    dino_keys = set(dino_df["Edge_Key"])

    cyano_df["Transfer_Match"] = cyano_df["Edge_Key"].apply(
        lambda x: 1 if x in dino_keys else 0
    )

    return cyano_df


def summarize(df, label):
    total = len(df)
    matches = int(df["Transfer_Match"].sum())
    rate = matches / total if total else 0

    return {
        "Analysis": label,
        "Total_Cyano_Edges": total,
        "Matched_Edges": matches,
        "Rate": rate
    }


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    cyano = pd.read_csv(CYANO_EDGE_FILE, encoding="utf-8-sig", low_memory=False)
    dino_post = pd.read_csv(DINO_POST2015_EDGE_FILE, encoding="utf-8-sig", low_memory=False)
    dino_all = pd.read_csv(DINO_ALL_EDGE_FILE, encoding="utf-8-sig", low_memory=False)

    cyano = clean_edges(cyano)
    dino_post = clean_edges(dino_post)
    dino_all = clean_edges(dino_all)

    print(f"Cyano edges: {len(cyano)}")
    print(f"Dino post-2015 edges: {len(dino_post)}")
    print(f"Dino all edges: {len(dino_all)}")
    print()

    # Predictive transfer (cyano → post-2015 dino)
    predictive = validate_transfer(cyano.copy(), dino_post)

    # Biological similarity (cyano → all dino)
    similarity = validate_transfer(cyano.copy(), dino_all)

    predictive.to_csv(outdir / "cyano_to_dino_post2015.csv", index=False)
    similarity.to_csv(outdir / "cyano_to_all_dino.csv", index=False)

    summary = pd.DataFrame([
        summarize(predictive, "Cyano → Post-2015 Dino"),
        summarize(similarity, "Cyano → All Dino"),
    ])

    summary.to_csv(outdir / "summary.csv", index=False)

    print("Transfer Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
