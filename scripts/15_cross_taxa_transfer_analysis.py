import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG_DIR = BASE / "FINAL_WORKSPACE/kg"
OUT_DIR = BASE / "FINAL_WORKSPACE/cross_taxa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DINO_PRE = KG_DIR / "dino_pre2016_semantic_edges.csv"
DINO_POST = KG_DIR / "dino_post2015_semantic_edges.csv"
CYANO = KG_DIR / "cyano_all_semantic_edges.csv"

OUT_SUMMARY = OUT_DIR / "cross_taxa_transfer_summary.csv"
OUT_CONSERVED = OUT_DIR / "cyano_all_vs_dino_all_conserved_edges.csv"
OUT_CONVERGENCE = OUT_DIR / "cyano_all_vs_dino_post2015_convergent_edges.csv"
OUT_TRANSFER = OUT_DIR / "cyano_plus_dino_pre2016_predicts_dino_post2015.csv"

def load_edges(path, label):
    df = pd.read_csv(path).fillna("")

    for col in ["source", "target", "source_type", "target_type", "relation", "weight"]:
        if col not in df.columns:
            df[col] = ""

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    df = df[(df["source"] != "") & (df["target"] != "")].copy()

    df["edge_key"] = df.apply(
        lambda r: "||".join(sorted([r["source"], r["target"]])),
        axis=1
    )

    df["dataset"] = label

    return df

def summarize_edges(df, weight_name):
    return df.groupby("edge_key", as_index=False).agg(
        source=("source", "first"),
        target=("target", "first"),
        source_type=("source_type", "first"),
        target_type=("target_type", "first"),
        relation=("relation", lambda x: ";".join(sorted(set(map(str, x))))),
        **{weight_name: ("weight", "sum")},
        support_datasets=("dataset", lambda x: ";".join(sorted(set(map(str, x)))))
    )

# ===============================
# LOAD
# ===============================

dino_pre = load_edges(DINO_PRE, "dino_pre2016")
dino_post = load_edges(DINO_POST, "dino_post2015")
cyano = load_edges(CYANO, "cyano_all")

dino_all = pd.concat([dino_pre, dino_post], ignore_index=True)
cyano_plus_dino_pre = pd.concat([cyano, dino_pre], ignore_index=True)

dino_all_g = summarize_edges(dino_all, "dino_all_weight")
dino_pre_g = summarize_edges(dino_pre, "dino_pre_weight")
dino_post_g = summarize_edges(dino_post, "dino_post_weight")
cyano_g = summarize_edges(cyano, "cyano_weight")
transfer_train_g = summarize_edges(cyano_plus_dino_pre, "transfer_train_weight")

# ===============================
# 1. CYANO_ALL VS DINO_ALL
# CONSERVED / SHARED BIOLOGY
# ===============================

conserved = cyano_g.merge(
    dino_all_g,
    on="edge_key",
    how="inner",
    suffixes=("_cyano", "_dino")
)

conserved["conservation_score"] = (
    conserved["cyano_weight"] +
    conserved["dino_all_weight"]
)

conserved = conserved.sort_values(
    by="conservation_score",
    ascending=False
)

conserved.to_csv(
    OUT_CONSERVED,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# 2. CYANO_ALL VS DINO_POST2015
# TEMPORAL CONVERGENCE
# ===============================

convergence = cyano_g.merge(
    dino_post_g,
    on="edge_key",
    how="inner",
    suffixes=("_cyano", "_dino_post")
)

convergence["convergence_score"] = (
    convergence["cyano_weight"] +
    convergence["dino_post_weight"]
)

convergence = convergence.sort_values(
    by="convergence_score",
    ascending=False
)

convergence.to_csv(
    OUT_CONVERGENCE,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# 3. TRANSFER VALIDATION
# CYANO_ALL + DINO_PRE2016 -> DINO_POST2015
# ===============================

train_keys = set(transfer_train_g["edge_key"])
future_keys = set(dino_post_g["edge_key"])
dino_pre_keys = set(dino_pre_g["edge_key"])
cyano_keys = set(cyano_g["edge_key"])

transfer_rows = []

for key in future_keys:

    in_dino_pre = key in dino_pre_keys
    in_cyano = key in cyano_keys
    in_transfer_train = key in train_keys

    if in_cyano and not in_dino_pre:
        transfer_type = "cyano_only_prior_signal"
    elif in_cyano and in_dino_pre:
        transfer_type = "cyano_and_dino_prior_signal"
    elif in_dino_pre and not in_cyano:
        transfer_type = "dino_prior_only_signal"
    else:
        transfer_type = "new_post2015_only"

    post_row = dino_post_g[dino_post_g["edge_key"] == key].iloc[0].to_dict()

    cyano_weight = 0
    dino_pre_weight = 0

    if in_cyano:
        cyano_weight = float(cyano_g.loc[cyano_g["edge_key"] == key, "cyano_weight"].iloc[0])

    if in_dino_pre:
        dino_pre_weight = float(dino_pre_g.loc[dino_pre_g["edge_key"] == key, "dino_pre_weight"].iloc[0])

    transfer_rows.append({
        "edge_key": key,
        "source": post_row.get("source", ""),
        "target": post_row.get("target", ""),
        "source_type": post_row.get("source_type", ""),
        "target_type": post_row.get("target_type", ""),
        "post2015_relation": post_row.get("relation", ""),
        "dino_post_weight": post_row.get("dino_post_weight", 0),
        "cyano_prior_weight": cyano_weight,
        "dino_pre_prior_weight": dino_pre_weight,
        "in_cyano_prior": int(in_cyano),
        "in_dino_pre_prior": int(in_dino_pre),
        "transfer_type": transfer_type,
        "transfer_support_score": cyano_weight + dino_pre_weight
    })

transfer_df = pd.DataFrame(transfer_rows)

transfer_df = transfer_df.sort_values(
    by=["transfer_support_score", "dino_post_weight"],
    ascending=False
)

transfer_df.to_csv(
    OUT_TRANSFER,
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# SUMMARY
# ===============================

summary = pd.DataFrame({
    "analysis": [
        "cyano_all_edges",
        "dino_pre2016_edges",
        "dino_post2015_edges",
        "dino_all_edges",
        "cyano_all_vs_dino_all_conserved_edges",
        "cyano_all_vs_dino_post2015_convergent_edges",
        "post2015_edges_with_cyano_prior_signal",
        "post2015_edges_with_dino_pre_prior_signal",
        "post2015_edges_with_cyano_only_prior_signal",
        "post2015_edges_new_only"
    ],
    "count": [
        len(cyano_g),
        len(dino_pre_g),
        len(dino_post_g),
        len(dino_all_g),
        len(conserved),
        len(convergence),
        int(transfer_df["in_cyano_prior"].sum()),
        int(transfer_df["in_dino_pre_prior"].sum()),
        int((transfer_df["transfer_type"] == "cyano_only_prior_signal").sum()),
        int((transfer_df["transfer_type"] == "new_post2015_only").sum())
    ]
})

summary.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== CROSS-TAXA TRANSFER ANALYSIS ==========")
print(summary.to_string(index=False))

print("\nSaved:")
print(OUT_SUMMARY)
print(OUT_CONSERVED)
print(OUT_CONVERGENCE)
print(OUT_TRANSFER)

print("\nTop conserved cyano_all vs dino_all edges:")
print(
    conserved[
        [
            "source_cyano",
            "target_cyano",
            "cyano_weight",
            "dino_all_weight",
            "conservation_score",
            "relation_cyano",
            "relation_dino"
        ]
    ].head(20).to_string(index=False)
)

print("\nTop cyano_all vs dino_post2015 convergent edges:")
print(
    convergence[
        [
            "source_cyano",
            "target_cyano",
            "cyano_weight",
            "dino_post_weight",
            "convergence_score",
            "relation_cyano",
            "relation_dino_post"
        ]
    ].head(20).to_string(index=False)
)

print("\nTransfer categories:")
print(transfer_df["transfer_type"].value_counts())

print("\nTop post-2015 dino edges with cyano/dino prior support:")
print(
    transfer_df[
        [
            "source",
            "target",
            "post2015_relation",
            "dino_post_weight",
            "cyano_prior_weight",
            "dino_pre_prior_weight",
            "transfer_type",
            "transfer_support_score"
        ]
    ].head(30).to_string(index=False)
)
