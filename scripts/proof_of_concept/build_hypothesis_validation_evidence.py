
#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

EDGE_FILE = PROJECT_ROOT / "FINAL_WORKSPACE" / "kg" / "dino_post2015_semantic_edges.csv"
HYPOTHESIS_FILE = PROJECT_ROOT / "FINAL_WORKSPACE" / "proof_of_concept" / "searchable_hypotheses.csv"
OUTPUT_FILE = PROJECT_ROOT / "FINAL_WORKSPACE" / "proof_of_concept" / "hypothesis_validation_evidence.csv"

def locate_metadata_file():
    candidates = [
        PROJECT_ROOT / "FINAL_WORKSPACE" / "splits" / "dino_post2015.csv",
        PROJECT_ROOT / "FINAL_WORKSPACE" / "splits" / "dino_post2015",
    ]
    for p in candidates:
        if p.is_file():
            return p
    split_dir = PROJECT_ROOT / "FINAL_WORKSPACE" / "splits"
    matches = sorted(split_dir.glob("dino_post2015*.csv")) if split_dir.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate FINAL_WORKSPACE/splits/dino_post2015.csv")

def canon(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().casefold().replace("_", " ")
    return re.sub(r"\s+", " ", s)

def split_docs(x):
    if x is None or pd.isna(x):
        return []
    out, seen = [], set()
    for part in re.split(r"[;,|]", str(x)):
        d = part.strip()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out

def first_col(df, names):
    return next((c for c in names if c in df.columns), None)

def main():
    metadata_file = locate_metadata_file()
    edges = pd.read_csv(EDGE_FILE)
    hyp = pd.read_csv(HYPOTHESIS_FILE)
    meta = pd.read_csv(metadata_file)

    for df in (edges, hyp, meta):
        df.columns = [str(c).strip() for c in df.columns]

    qcol = first_col(hyp, ["Query_Entity", "Query_Entity_Normalized", "Source"])
    pcol = first_col(hyp, ["Predicted_Entity", "Target"])
    vcol = first_col(hyp, ["Validation_Status", "Temporal_Validation_Status"])
    ccol = first_col(hyp, ["Hypothesis_Class", "Hypothesis_Type"])
    scol = first_col(hyp, ["AI_Score", "Final_AI_Score", "Score"])
    dcol = first_col(meta, ["document_id", "Document_ID", "paper_id"])

    if not qcol or not pcol or not dcol:
        raise ValueError("Could not identify required hypothesis or metadata columns.")

    if vcol:
        hyp = hyp[hyp[vcol].astype(str).str.strip().str.casefold().eq("validated")].copy()

    edges["_pair"] = edges.apply(
        lambda r: "||".join(sorted([canon(r["source"]), canon(r["target"])])),
        axis=1,
    )
    hyp["_pair"] = hyp.apply(
        lambda r: "||".join(sorted([canon(r[qcol]), canon(r[pcol])])),
        axis=1,
    )

    meta[dcol] = meta[dcol].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[dcol]).set_index(dcol)

    rows = []
    for _, h in hyp.iterrows():
        matches = edges[edges["_pair"].eq(h["_pair"])]
        if matches.empty:
            rows.append({
                "Query_Entity": h[qcol],
                "Predicted_Entity": h[pcol],
                "Hypothesis_Class": h.get(ccol, "") if ccol else "",
                "AI_Score": h.get(scol, "") if scol else "",
                "Validation_Status": h.get(vcol, "Validated") if vcol else "Validated",
                "Evidence_Match": "NO_MATCHING_POST2015_EDGE",
            })
            continue

        for _, e in matches.iterrows():
            docs = split_docs(e.get("support_documents", ""))
            for doc_id in docs:
                rec = {
                    "Query_Entity": h[qcol],
                    "Predicted_Entity": h[pcol],
                    "Hypothesis_Class": h.get(ccol, "") if ccol else "",
                    "AI_Score": h.get(scol, "") if scol else "",
                    "Validation_Status": h.get(vcol, "Validated") if vcol else "Validated",
                    "Edge_Source": e.get("source", ""),
                    "Edge_Target": e.get("target", ""),
                    "Edge_Relation": e.get("relation", ""),
                    "Edge_Weight": e.get("weight", ""),
                    "Support_Count": len(docs),
                    "First_Year": e.get("first_year", ""),
                    "Last_Year": e.get("last_year", ""),
                    "Document_ID": doc_id,
                    "Evidence_Match": "MATCHED",
                }
                if doc_id in meta.index:
                    m = meta.loc[doc_id]
                    rec.update({
                        "Title": m.get("title", ""),
                        "Journal": m.get("journal", ""),
                        "Year": m.get("year", ""),
                        "DOI": m.get("doi_clean", "") or m.get("doi", ""),
                        "PMID": m.get("pmid_clean", "") or m.get("pmid", ""),
                        "URL": m.get("url", ""),
                        "Source_DB": m.get("source_database", "") or m.get("source_db", ""),
                    })
                else:
                    rec.update({
                        "Title": "", "Journal": "", "Year": "", "DOI": "",
                        "PMID": "", "URL": "", "Source_DB": "",
                        "Evidence_Match": "DOCUMENT_ID_NOT_FOUND",
                    })
                rows.append(rec)

    out = pd.DataFrame(rows)

    if not out.empty:
        dedup_cols = [c for c in ["Query_Entity","Predicted_Entity","Document_ID","Edge_Relation"] if c in out.columns]
        out = out.drop_duplicates(subset=dedup_cols)
        if "Year" in out.columns:
            out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
            out = out.sort_values(
                ["Query_Entity","Predicted_Entity","Year"],
                ascending=[True, True, False],
                na_position="last"
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("="*72)
    print("Created:", OUTPUT_FILE)
    print("Rows:", len(out))
    if not out.empty:
        print("Unique validated hypotheses:", out[["Query_Entity","Predicted_Entity"]].drop_duplicates().shape[0])
        if "Document_ID" in out.columns:
            print("Unique supporting documents:", out["Document_ID"].replace("", pd.NA).dropna().nunique())
        print("\nPreview:")
        cols = [c for c in ["Query_Entity","Predicted_Entity","Document_ID","Title","Journal","Year","DOI"] if c in out.columns]
        print(out[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
