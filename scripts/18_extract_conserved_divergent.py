from pathlib import Path
import pandas as pd


INPUT_FILE = "results/cyano_transfer/cyano_to_dino_post2015.csv"
OUTPUT_DIR = "results/cyano_transfer/biological_insights"


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # -----------------------------
    # CONSERVED (matched)
    # -----------------------------
    conserved = df[df["Transfer_Match"] == 1].copy()
    conserved = conserved.sort_values(["Weight", "Paper_Count"], ascending=False)

    conserved.to_csv(outdir / "conserved_stx_relationships.csv", index=False)

    # -----------------------------
    # DIVERGENT (not matched)
    # -----------------------------
    divergent = df[df["Transfer_Match"] == 0].copy()
    divergent = divergent.sort_values(["Weight", "Paper_Count"], ascending=False)

    divergent.to_csv(outdir / "divergent_cyano_only_relationships.csv", index=False)

    # -----------------------------
    # TOP TABLES (for paper)
    # -----------------------------
    top_conserved = conserved.head(30)
    top_divergent = divergent.head(30)

    top_conserved.to_csv(outdir / "top_conserved.csv", index=False)
    top_divergent.to_csv(outdir / "top_divergent.csv", index=False)

    # -----------------------------
    # CATEGORY SUMMARY
    # -----------------------------
    def categorize(row):
        src = str(row["Source"]).lower()
        tgt = str(row["Target"]).lower()

        if "sxt" in src or "sxt" in tgt:
            return "Gene-related"
        if any(x in src+tgt for x in ["nitrogen", "phosphate", "temperature", "light", "ph"]):
            return "Environmental"
        if any(x in src+tgt for x in ["biosynthesis", "pathway", "metabolite"]):
            return "Mechanistic"
        if any(x in src+tgt for x in ["evolution", "transfer", "phylogeny"]):
            return "Evolutionary"
        return "Other"

    df["Category"] = df.apply(categorize, axis=1)

    summary = df.groupby(["Transfer_Match", "Category"]).size().reset_index(name="Count")

    summary.to_csv(outdir / "category_summary.csv", index=False)

    print("Saved outputs in:", outdir)
    print()
    print("Top conserved:")
    print(top_conserved[["Source","Target","Relation","Weight"]].head(10).to_string(index=False))
    print()
    print("Top divergent:")
    print(top_divergent[["Source","Target","Relation","Weight"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
