import pandas as pd
from pathlib import Path


HYP_FILE = "results/combined/STX_COMBINED_HYPOTHESES.csv"
DINO_POST = "../New/data/graphs_semantic/semantic_post2015_edges_filtered.csv"

OUTPUT_DIR = "results/combined/validation"


def edge_key(a, b):
    return tuple(sorted([str(a).lower(), str(b).lower()]))


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    hyp = pd.read_csv(HYP_FILE, encoding="utf-8-sig")
    dino = pd.read_csv(DINO_POST, encoding="utf-8-sig", low_memory=False)

    print(f"Hypotheses: {len(hyp)}")
    print(f"Dino post-2015 edges: {len(dino)}")

    # Build lookup set
    dino_keys = set(
        edge_key(r["Source"], r["Target"]) for _, r in dino.iterrows()
    )

    # Validate
    hyp["Match"] = hyp.apply(
        lambda r: 1 if edge_key(r["Source"], r["Target"]) in dino_keys else 0,
        axis=1
    )

    total = len(hyp)
    matches = int(hyp["Match"].sum())
    precision = matches / total if total else 0

    print("\n=== Combined Model Results ===")
    print(f"Total hypotheses: {total}")
    print(f"Matches in post-2015 dino: {matches}")
    print(f"Precision: {precision:.4f}")

    # =========================
    # SAVE FULL RESULTS
    # =========================
    hyp.to_csv(outdir / "combined_hypotheses_with_validation.csv", index=False)

    # =========================
    # SAVE SUMMARY
    # =========================
    summary = pd.DataFrame([{
        "Model": "Combined (Cyano + Pre-2016 Dino)",
        "Total_Hypotheses": total,
        "Matches": matches,
        "Precision": precision
    }])

    summary.to_csv(outdir / "combined_validation_summary.csv", index=False)

    # =========================
    # SAVE TOP MATCHES
    # =========================
    top_matches = (
        hyp[hyp["Match"] == 1]
        .sort_values("Score", ascending=False)
        .head(50)
    )

    top_matches.to_csv(outdir / "top_matched_hypotheses.csv", index=False)

    # =========================
    # SAVE TOP FAILURES
    # =========================
    top_failures = (
        hyp[hyp["Match"] == 0]
        .sort_values("Score", ascending=False)
        .head(50)
    )

    top_failures.to_csv(outdir / "top_unmatched_hypotheses.csv", index=False)

    print("\nSaved files:")
    print(outdir / "combined_hypotheses_with_validation.csv")
    print(outdir / "combined_validation_summary.csv")
    print(outdir / "top_matched_hypotheses.csv")
    print(outdir / "top_unmatched_hypotheses.csv")


if __name__ == "__main__":
    main()
