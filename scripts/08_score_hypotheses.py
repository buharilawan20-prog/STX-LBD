from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


HYPOTHESIS_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES.csv"
EMBEDDING_FILE = "results/embeddings/STX_NODE_EMBEDDINGS.csv"
OUTPUT_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES_with_embeddings.csv"


def cosine_similarity(v1, v2) -> float:
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


def load_embeddings(path: str) -> dict:
    emb_df = pd.read_csv(path, encoding="utf-8-sig")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]

    return {
        str(row["Node"]): row[emb_cols].values.astype(float)
        for _, row in emb_df.iterrows()
    }


def minmax_scale(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def main():
    hyp_path = Path(HYPOTHESIS_FILE)
    emb_path = Path(EMBEDDING_FILE)

    if not hyp_path.exists():
        raise FileNotFoundError(f"Missing hypothesis file: {HYPOTHESIS_FILE}")

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {EMBEDDING_FILE}")

    hyp_df = pd.read_csv(hyp_path, encoding="utf-8-sig")
    embeddings = load_embeddings(EMBEDDING_FILE)

    source_target_scores = []
    bridge_scores = []

    for _, row in hyp_df.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])

        # Source-target similarity
        if source in embeddings and target in embeddings:
            st_score = cosine_similarity(embeddings[source], embeddings[target])
        else:
            st_score = np.nan

        source_target_scores.append(st_score)

        # Bridge-aware similarity
        bridges_raw = str(row.get("Bridge_Nodes", ""))
        bridges = [b.strip() for b in bridges_raw.split(";") if b.strip()]

        bridge_similarities = []

        for bridge in bridges:
            if source in embeddings and bridge in embeddings:
                bridge_similarities.append(
                    cosine_similarity(embeddings[source], embeddings[bridge])
                )

            if target in embeddings and bridge in embeddings:
                bridge_similarities.append(
                    cosine_similarity(embeddings[target], embeddings[bridge])
                )

        if bridge_similarities:
            bridge_score = float(np.mean(bridge_similarities))
        else:
            bridge_score = np.nan

        bridge_scores.append(bridge_score)

    hyp_df["Embedding_Source_Target_Similarity"] = source_target_scores
    hyp_df["Embedding_Bridge_Mean_Similarity"] = bridge_scores

    # Normalize scoring components
    hyp_df["Structural_Score_Norm"] = minmax_scale(hyp_df["Structural_Score"])
    hyp_df["Embedding_ST_Norm"] = minmax_scale(
        hyp_df["Embedding_Source_Target_Similarity"]
    )
    hyp_df["Embedding_Bridge_Norm"] = minmax_scale(
        hyp_df["Embedding_Bridge_Mean_Similarity"]
    )

    # Integrated score
    hyp_df["Embedding_Integrated_Score"] = (
        0.50 * hyp_df["Structural_Score_Norm"]
        + 0.30 * hyp_df["Embedding_ST_Norm"]
        + 0.20 * hyp_df["Embedding_Bridge_Norm"]
    )

    hyp_df = hyp_df.sort_values(
        "Embedding_Integrated_Score",
        ascending=False
    )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hyp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved scored hypotheses: {output_path}")
    print(f"Total hypotheses scored: {len(hyp_df)}")
    print()
    print("Top AI-ranked hypotheses:")
    preview_cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Structural_Score",
        "Embedding_Source_Target_Similarity",
        "Embedding_Bridge_Mean_Similarity",
        "Embedding_Integrated_Score",
        "Bridge_Nodes",
    ]

    print(hyp_df[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
