import pandas as pd
import networkx as nx
from pathlib import Path

from node2vec import Node2Vec

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_pre2016_semantic_edges.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_OUT = OUT_DIR / "dino_pre2016_node2vec_embeddings.csv"
MODEL_OUT = OUT_DIR / "dino_pre2016_node2vec_model.model"

# ===============================
# NODE2VEC SETTINGS
# ===============================

DIMENSIONS = 64
WALK_LENGTH = 20
NUM_WALKS = 200
WINDOW = 5
MIN_COUNT = 1
BATCH_WORDS = 4
WORKERS = 2
P = 1
Q = 1

# ===============================
# LOAD EDGES
# ===============================

df = pd.read_csv(EDGE_FILE).fillna("")

for col in ["source", "target", "weight"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["weight"] = pd.to_numeric(
    df["weight"],
    errors="coerce"
).fillna(1)

# ===============================
# BUILD GRAPH
# ===============================

G = nx.Graph()

for _, row in df.iterrows():

    s = str(row["source"]).strip()
    t = str(row["target"]).strip()

    if not s or not t or s == t:
        continue

    w = float(row["weight"])

    if G.has_edge(s, t):
        G[s][t]["weight"] += w
    else:
        G.add_edge(s, t, weight=w)

print("\n========== NODE2VEC TRAINING ==========")
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

# ===============================
# TRAIN NODE2VEC
# ===============================

node2vec = Node2Vec(
    G,
    dimensions=DIMENSIONS,
    walk_length=WALK_LENGTH,
    num_walks=NUM_WALKS,
    workers=WORKERS,
    p=P,
    q=Q,
    weight_key="weight",
    seed=42
)

model = node2vec.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS,
    seed=42
)

# ===============================
# SAVE MODEL
# ===============================

model.save(str(MODEL_OUT))

# ===============================
# SAVE EMBEDDINGS
# ===============================

rows = []

for node in model.wv.index_to_key:
    vec = model.wv[node]

    row = {"node": node}

    for i, value in enumerate(vec):
        row[f"emb_{i}"] = float(value)

    rows.append(row)

emb_df = pd.DataFrame(rows)

emb_df.to_csv(
    EMBEDDING_OUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved embeddings:")
print(EMBEDDING_OUT)

print("\nSaved model:")
print(MODEL_OUT)

print("\nEmbedding matrix:")
print(emb_df.shape)

print("\nTop nodes saved:")
print(emb_df["node"].head(20).to_string(index=False))
