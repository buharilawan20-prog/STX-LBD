from pathlib import Path
import pandas as pd


INPUT_FILE = "results/semantic_hypotheses/STX_SEMANTIC_HYPOTHESES.csv"
OUTPUT_DIR = "results/semantic_hypotheses"

OUTPUT_ALL = "STX_SEMANTIC_HYPOTHESES_PRIORITY_ALL.csv"
OUTPUT_TOP = "STX_SEMANTIC_HYPOTHESES_PRIORITY_TOP500.csv"


HIGH_PRIORITY_TYPES = {
    "GENE_TOXIN_HYPOTHESIS",
    "GENE_TOXIN_PHENOTYPE_HYPOTHESIS",
    "GENE_MECHANISM_HYPOTHESIS",
    "GENE_REGULATION_HYPOTHESIS",
    "GENE_PRESENCE_ABSENCE_HYPOTHESIS",
    "GENE_EVOLUTION_HYPOTHESIS",
    "GENE_DOMAIN_HYPOTHESIS",
    "GENE_MECHANISM_PHRASE_HYPOTHESIS",
    "GENE_REGULATION_PHRASE_HYPOTHESIS",
    "GENE_EVOLUTION_PHRASE_HYPOTHESIS",
    "GENE_BIOSYNTHESIS_PHRASE_HYPOTHESIS",
    "ENV_GENE_HYPOTHESIS",
    "ENV_MECHANISM_HYPOTHESIS",
    "ENV_REGULATION_HYPOTHESIS",
    "ENV_TOXIN_HYPOTHESIS",
    "ENV_TOXIN_PHENOTYPE_HYPOTHESIS",
    "EVOLUTION_GENE_PRESENCE_HYPOTHESIS",
    "EVOLUTION_GENE_MECHANISM_HYPOTHESIS",
    "EVOLUTION_MECHANISM_HYPOTHESIS",
    "REGULATION_TOXIN_PHENOTYPE_HYPOTHESIS",
    "MECHANISM_TOXIN_PHENOTYPE_HYPOTHESIS",
    "SPECIES_GENE_HYPOTHESIS",
    "SPECIES_GENE_PRESENCE_ABSENCE_HYPOTHESIS",
    "SPECIES_REGULATION_HYPOTHESIS",
    "SPECIES_MECHANISM_HYPOTHESIS",
    "SPECIES_EVOLUTION_HYPOTHESIS",
}

DOWNRANK_TYPES = {
    "SPECIES_TOXIN_PHENOTYPE_HYPOTHESIS",
}

REMOVE_PATTERNS = {
    "Alexandrium — paralytic shellfish toxins",
    "Alexandrium — saxitoxin",
    "Gymnodinium — saxitoxin",
    "dinoflagellates — saxitoxin",
}

GENE_KEYWORDS = [
    "sxt", "sxtA", "sxtA1", "sxtA4", "sxtG", "sxtB", "sxtU",
    "gene", "genes", "homolog", "homologs",
]

MECHANISM_KEYWORDS = [
    "biosynthesis", "toxin production", "toxin content", "toxin profile",
    "toxin composition", "gene expression", "transcription", "regulation",
    "enzyme", "activity", "pathway",
]

EVOLUTION_KEYWORDS = [
    "evolution", "phylogeny", "phylogenetic", "divergence", "conserved",
    "horizontal gene transfer", "hgt", "independent acquisition",
    "gene loss", "gene duplication", "presence", "absence",
]

ENV_KEYWORDS = [
    "temperature", "salinity", "nitrogen", "nitrate", "phosphate",
    "phosphorus", "nutrient", "limitation", "stress", "light",
]

LOW_VALUE_NODES = {
    "saxitoxin",
    "paralytic shellfish toxins",
    "toxic",
    "non-toxic",
    "toxin profile",
    "toxin profiles",
}


def text_blob(row):
    return (
        f"{row.get('Source','')} {row.get('Target','')} "
        f"{row.get('Hypothesis_Type','')} {row.get('Bridge_Nodes','')} "
        f"{row.get('Bridge_Relation_Hints','')}"
    ).lower()


def contains_any(text, keywords):
    return any(k.lower() in text for k in keywords)


def priority_bonus(row):
    text = text_blob(row)
    bonus = 0.0

    if row["Hypothesis_Type"] in HIGH_PRIORITY_TYPES:
        bonus += 5.0

    if row["Hypothesis_Type"] in DOWNRANK_TYPES:
        bonus -= 5.0

    if contains_any(text, GENE_KEYWORDS):
        bonus += 3.0

    if contains_any(text, MECHANISM_KEYWORDS):
        bonus += 2.5

    if contains_any(text, EVOLUTION_KEYWORDS):
        bonus += 2.5

    if contains_any(text, ENV_KEYWORDS):
        bonus += 2.0

    # Reward explicit relation hints
    hints = str(row.get("Bridge_Relation_Hints", "")).lower()
    if "present_in" in hints or "absent_in" in hints:
        bonus += 2.0
    if "regulates" in hints or "involved_in" in hints:
        bonus += 2.0
    if "evolutionary_link" in hints:
        bonus += 2.0
    if "increases" in hints or "decreases" in hints:
        bonus += 1.5

    # Penalize if both source and target are overly broad
    source = str(row["Source"]).strip().lower()
    target = str(row["Target"]).strip().lower()
    if source in LOW_VALUE_NODES and target in LOW_VALUE_NODES:
        bonus -= 8.0

    return bonus


def biological_category(row):
    h = row["Hypothesis_Type"]

    if "EVOLUTION" in h:
        return "Evolutionary mechanism"
    if "GENE" in h and "REGULATION" in h:
        return "Gene regulation"
    if "GENE" in h and "PRESENCE" in h:
        return "Gene presence/absence"
    if "GENE" in h and "TOXIN" in h:
        return "Gene-toxin biology"
    if "ENV" in h:
        return "Environmental driver"
    if "MECHANISM" in h:
        return "Mechanistic biology"
    if "SPECIES_GENE" in h:
        return "Species-gene link"
    return "Other"


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    df["Priority_Bonus"] = df.apply(priority_bonus, axis=1)

    df["Priority_Score"] = (
        pd.to_numeric(df["Structural_Score"], errors="coerce").fillna(0)
        + df["Priority_Bonus"]
    )

    df["Biological_Category"] = df.apply(biological_category, axis=1)

    # Remove clearly low-value broad species-toxin phenotype dominance only if weak
    df = df[
        ~(
            (df["Hypothesis_Type"].isin(DOWNRANK_TYPES))
            & (df["Priority_Score"] < df["Priority_Score"].quantile(0.75))
        )
    ].copy()

    df = df.sort_values("Priority_Score", ascending=False)

    all_out = outdir / OUTPUT_ALL
    top_out = outdir / OUTPUT_TOP

    df.to_csv(all_out, index=False, encoding="utf-8-sig")
    df.head(500).to_csv(top_out, index=False, encoding="utf-8-sig")

    print(f"Saved priority hypotheses: {all_out}")
    print(f"Saved top 500: {top_out}")
    print()
    print(f"Remaining hypotheses: {len(df)}")
    print()
    print("Hypothesis type counts:")
    print(df["Hypothesis_Type"].value_counts().head(30))
    print()
    print("Biological category counts:")
    print(df["Biological_Category"].value_counts())
    print()
    print("Top priority hypotheses:")
    cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Biological_Category",
        "Structural_Score",
        "Priority_Bonus",
        "Priority_Score",
        "Bridge_Nodes",
        "Bridge_Relation_Hints",
    ]
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
