from pathlib import Path
import pandas as pd


INPUT_FILE = "results/semantic_hypotheses/STX_SEMANTIC_HYPOTHESES_PRIORITY_ALL.csv"
OUTPUT_DIR = "results/semantic_hypotheses"

OUTPUT_FILE = "STX_DISCOVERY_SHORTLIST_BY_CATEGORY.csv"
SUMMARY_FILE = "STX_DISCOVERY_SHORTLIST_SUMMARY.csv"

TOP_N_PER_CATEGORY = 50


TARGET_CATEGORIES = [
    "Gene presence/absence",
    "Gene regulation",
    "Evolutionary mechanism",
    "Environmental driver",
    "Gene-toxin biology",
    "Mechanistic biology",
    "Species-gene link",
]


EXCLUDE_BROAD_TERMS = {
    "saxitoxin",
    "paralytic shellfish toxins",
    "toxic",
    "non-toxic",
    "toxin profile",
    "toxin profiles",
    "toxin content",
    "toxin contents",
}


MECHANISTIC_KEYWORDS = [
    "sxt",
    "sxtA",
    "sxtA1",
    "sxtA4",
    "sxtG",
    "sxtB",
    "sxtU",
    "gene expression",
    "biosynthesis",
    "regulation",
    "presence",
    "absence",
    "loss",
    "duplication",
    "homolog",
    "phylogeny",
    "evolution",
    "divergence",
    "conserved",
    "hgt",
    "horizontal gene transfer",
    "temperature",
    "salinity",
    "nitrogen",
    "nitrate",
    "phosphate",
    "nutrient",
    "limitation",
]


def has_mechanistic_signal(row) -> bool:
    text = (
        f"{row.get('Source','')} {row.get('Target','')} "
        f"{row.get('Bridge_Nodes','')} {row.get('Bridge_Relation_Hints','')}"
    ).lower()

    return any(k.lower() in text for k in MECHANISTIC_KEYWORDS)


def is_too_broad(row) -> bool:
    source = str(row["Source"]).strip().lower()
    target = str(row["Target"]).strip().lower()

    return source in EXCLUDE_BROAD_TERMS and target in EXCLUDE_BROAD_TERMS


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    df = df.copy()

    # Basic cleanup
    df = df[~df.apply(is_too_broad, axis=1)].copy()

    # Keep hypotheses with mechanistic signal
    df = df[df.apply(has_mechanistic_signal, axis=1)].copy()

    shortlisted = []

    for category in TARGET_CATEGORIES:
        cat_df = df[df["Biological_Category"] == category].copy()

        if cat_df.empty:
            continue

        cat_df = cat_df.sort_values("Priority_Score", ascending=False)
        cat_df = cat_df.head(TOP_N_PER_CATEGORY)
        shortlisted.append(cat_df)

    if not shortlisted:
        print("No hypotheses passed shortlist filtering.")
        return

    short_df = pd.concat(shortlisted, ignore_index=True)

    # Add simple interpretation flags
    short_df["Has_STX_Gene_Signal"] = short_df.apply(
        lambda r: any(x in f"{r['Source']} {r['Target']} {r['Bridge_Nodes']}".lower()
                      for x in ["sxt", "sxta", "sxtg", "sxtb", "sxtu"]),
        axis=1,
    )

    short_df["Has_Environment_Signal"] = short_df.apply(
        lambda r: any(x in f"{r['Source']} {r['Target']} {r['Bridge_Nodes']}".lower()
                      for x in ["temperature", "salinity", "nitrogen", "nitrate", "phosphate", "nutrient"]),
        axis=1,
    )

    short_df["Has_Evolution_Signal"] = short_df.apply(
        lambda r: any(x in f"{r['Source']} {r['Target']} {r['Bridge_Nodes']}".lower()
                      for x in ["evolution", "phylogen", "divergence", "conserved", "hgt", "horizontal gene transfer"]),
        axis=1,
    )

    short_df["Has_Regulation_Signal"] = short_df.apply(
        lambda r: any(x in f"{r['Source']} {r['Target']} {r['Bridge_Nodes']}".lower()
                      for x in ["expression", "regulation", "transcription", "up-regulated", "down-regulated"]),
        axis=1,
    )

    short_df = short_df.sort_values(
        ["Biological_Category", "Priority_Score"],
        ascending=[True, False],
    )

    output_path = outdir / OUTPUT_FILE
    short_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = (
        short_df.groupby("Biological_Category")
        .agg(
            Count=("Source", "count"),
            Mean_Priority_Score=("Priority_Score", "mean"),
            Max_Priority_Score=("Priority_Score", "max"),
            STX_Gene_Signal=("Has_STX_Gene_Signal", "sum"),
            Environment_Signal=("Has_Environment_Signal", "sum"),
            Evolution_Signal=("Has_Evolution_Signal", "sum"),
            Regulation_Signal=("Has_Regulation_Signal", "sum"),
        )
        .reset_index()
    )

    summary_path = outdir / SUMMARY_FILE
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved discovery shortlist: {output_path}")
    print(f"Saved summary: {summary_path}")
    print()
    print("Shortlist summary:")
    print(summary.to_string(index=False))
    print()
    print("Top discovery hypotheses by category:")
    cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Biological_Category",
        "Priority_Score",
        "Has_STX_Gene_Signal",
        "Has_Environment_Signal",
        "Has_Evolution_Signal",
        "Has_Regulation_Signal",
        "Bridge_Nodes",
        "Bridge_Relation_Hints",
    ]

    for category in TARGET_CATEGORIES:
        cat_df = short_df[short_df["Biological_Category"] == category]
        if len(cat_df) == 0:
            continue

        print("\n###", category)
        print(cat_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
