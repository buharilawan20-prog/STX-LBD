from pathlib import Path
from itertools import combinations
import math
import pandas as pd
import networkx as nx


# =====================================================
# INPUT / OUTPUT
# =====================================================
EDGE_FILE = "data/graphs_dino/dino_semantic_edges_normalized_collapsed.csv"

OUTDIR = Path("results/dino_temporal_normalized_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)

YEAR_SPLIT = 2015
TOP_N = 5000


# =====================================================
# BIOLOGICAL TERMS
# =====================================================
CORE_TERMS = [
    "saxitoxin",
    "paralytic shellfish toxins",
    "sxtA",
    "sxtG",
    "sxtI",
    "sxt genes",
    "saxitoxin biosynthesis",
    "biosynthesis",
    "toxin production",
    "toxin profile",
    "toxin content",
    "temperature",
    "salinity",
    "nitrogen",
    "phosphorus",
    "light",
    "oxidative stress",
    "gene expression",
    "phylogeny",
    "evolution",
    "horizontal gene transfer",
    "gene transfer",
    "Alexandrium",
    "Gymnodinium catenatum",
]


NOISY_TERMS = [
    "has",
    "of g",
    "profiles of",
    "toxin contents",
    "toxins c",
    "dinoflagellate gymnodinium",
    "dinoflagellate alexandrium",
]


# =====================================================
# FUNCTIONS
# =====================================================
def parse_years(x):
    years = []
    for y in str(x).split(";"):
        try:
            years.append(int(y))
        except:
            pass
    return years


def edge_key(a, b):
    return tuple(sorted([
        str(a).strip().lower(),
        str(b).strip().lower()
    ]))


def is_noisy(x):
    x = str(x).lower()
    return any(n in x for n in NOISY_TERMS)


def biological_bonus(u, v, bridges):
    text = " ".join([str(u), str(v), " ".join(bridges)]).lower()

    bonus = 0

    for term in CORE_TERMS:
        if term.lower() in text:
            bonus += 1

    # STX-centered
    if "saxitoxin" in text or "stx" in text:
        bonus += 5

    # gene + environment
    gene_terms = ["sxta", "sxtg", "sxti", "sxt genes"]
    env_terms = ["temperature", "salinity", "nitrogen", "phosphorus", "light"]

    if any(g in text for g in gene_terms) and any(e in text for e in env_terms):
        bonus += 8

    # biosynthesis + toxin
    if "biosynthesis" in text and ("toxin" in text or "saxitoxin" in text):
        bonus += 6

    # evolution + gene/toxin
    if ("gene transfer" in text or "phylogeny" in text or "evolution" in text) and (
        "sxt" in text or "saxitoxin" in text or "toxin" in text
    ):
        bonus += 6

    return bonus


def split_temporal_edges(df):
    train_rows = []
    test_rows = []

    for _, row in df.iterrows():
        years = parse_years(row["Years"])

        if not years:
            continue

        # earlier stronger logic:
        # train if relationship existed at or before 2015
        # test if relationship appeared after 2015
        if any(y <= YEAR_SPLIT for y in years):
            train_rows.append(row)

        if any(y > YEAR_SPLIT for y in years):
            test_rows.append(row)

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        s = str(row["Source"])
        t = str(row["Target"])

        if is_noisy(s) or is_noisy(t):
            continue

        weight = float(row.get("Weight", 1))

        G.add_edge(
            s,
            t,
            weight=weight,
            relation=row.get("Relation", "")
        )

    return G


def weighted_common_neighbor_score(G, u, v, common):
    score = 0

    for z in common:
        w1 = G[u][z].get("weight", 1)
        w2 = G[v][z].get("weight", 1)
        score += min(w1, w2)

    return score


def adamic_adar_score(G, common):
    score = 0

    for z in common:
        deg = G.degree(z)
        if deg > 1:
            score += 1 / math.log(deg)

    return score


def generate_hypotheses(G):
    rows = []

    nodes = list(G.nodes())

    for u, v in combinations(nodes, 2):
        if G.has_edge(u, v):
            continue

        if is_noisy(u) or is_noisy(v):
            continue

        common = list(nx.common_neighbors(G, u, v))

        if len(common) == 0:
            continue

        cn = len(common)
        wcn = weighted_common_neighbor_score(G, u, v, common)
        aa = adamic_adar_score(G, common)
        pa = G.degree(u) * G.degree(v)
        bio = biological_bonus(u, v, common)

        # earlier-style structural + biological priority score
        score = (
            (cn * 2.0)
            + (wcn * 0.5)
            + (aa * 3.0)
            + (math.log1p(pa) * 0.5)
            + (bio * 2.0)
        )

        rows.append({
            "Source": u,
            "Target": v,
            "Score": score,
            "Common_Neighbors": cn,
            "Weighted_Common_Neighbors": wcn,
            "Adamic_Adar": aa,
            "Preferential_Attachment": pa,
            "Bio_Bonus": bio,
            "Bridge_Nodes": ";".join(sorted(common[:25]))
        })

    hyp = pd.DataFrame(rows)

    if hyp.empty:
        return hyp

    hyp = hyp.sort_values(
        ["Score", "Bio_Bonus", "Weighted_Common_Neighbors", "Common_Neighbors"],
        ascending=False
    ).head(TOP_N)

    return hyp


def validate_hypotheses(hyp, test_df):
    test_keys = set(
        test_df.apply(lambda r: edge_key(r["Source"], r["Target"]), axis=1)
    )

    hyp["Match"] = hyp.apply(
        lambda r: 1 if edge_key(r["Source"], r["Target"]) in test_keys else 0,
        axis=1
    )

    return hyp


def precision_at_k(df, k):
    sub = df.head(k)
    if len(sub) == 0:
        return 0
    return sub["Match"].sum() / len(sub)


def hits_at_k(df, k):
    return int(df.head(k)["Match"].sum())


# =====================================================
# MAIN
# =====================================================
def main():
    print("\nLoading normalized collapsed dino edges...")
    df = pd.read_csv(EDGE_FILE, low_memory=False)

    print(f"Total normalized dino edges: {len(df)}")

    train_df, test_df = split_temporal_edges(df)

    print(f"Train edges with evidence ≤ {YEAR_SPLIT}: {len(train_df)}")
    print(f"Test edges with evidence > {YEAR_SPLIT}: {len(test_df)}")

    train_df.to_csv(OUTDIR / "dino_train_pre2016_edges.csv", index=False)
    test_df.to_csv(OUTDIR / "dino_test_post2015_edges.csv", index=False)

    print("\nBuilding pre-2016 dino graph...")
    G = build_graph(train_df)

    print(f"Train graph nodes: {G.number_of_nodes()}")
    print(f"Train graph edges: {G.number_of_edges()}")

    print("\nGenerating hypotheses with earlier-style structural + biological scoring...")
    hyp = generate_hypotheses(G)

    print(f"Hypotheses generated: {len(hyp)}")

    print("\nValidating against post-2015 dino edges...")
    hyp = validate_hypotheses(hyp, test_df)

    hyp.to_csv(OUTDIR / "normalized_dino_hypotheses_ranked_validated_v2.csv", index=False)

    metrics = []
    for k in [10, 20, 50, 100, 200, 500, 1000]:
        metrics.append({
            "K": k,
            "Hits@K": hits_at_k(hyp, k),
            "Precision@K": precision_at_k(hyp, k)
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTDIR / "normalized_dino_temporal_validation_metrics_v2.csv", index=False)

    matched = hyp[hyp["Match"] == 1].copy()
    unmatched = hyp[hyp["Match"] == 0].copy()

    matched.to_csv(OUTDIR / "top_validated_dino_hypotheses_v2.csv", index=False)
    unmatched.to_csv(OUTDIR / "top_unmatched_dino_hypotheses_v2.csv", index=False)

    print("\nTemporal validation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop validated hypotheses:")
    print(
        matched[
            [
                "Source",
                "Target",
                "Score",
                "Common_Neighbors",
                "Weighted_Common_Neighbors",
                "Bio_Bonus"
            ]
        ].head(30).to_string(index=False)
    )

    print("\nSaved outputs in:")
    print(OUTDIR)


if __name__ == "__main__":
    main()
