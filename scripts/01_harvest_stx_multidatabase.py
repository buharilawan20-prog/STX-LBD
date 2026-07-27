import time
import requests
import pandas as pd
from pathlib import Path
from Bio import Entrez

# =========================
# SETTINGS
# =========================
Entrez.email = "buharilawan20@gmail.com"
Entrez.tool = "STX_LBD_Corpus_Enrichment"

OUTDIR = Path("data/raw")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_PUBMED = 1000
MAX_OPENALEX = 200
MAX_CROSSREF = 100

HEADERS = {
    "User-Agent": "STX-LBD-Corpus-Enrichment/1.0 (mailto:buharilawan20@gmail.com)"
}

DINO_QUERIES = [
    "saxitoxin dinoflagellate",
    "paralytic shellfish toxins dinoflagellate",
    "Alexandrium saxitoxin",
    "Gymnodinium catenatum saxitoxin",
    "Pyrodinium bahamense saxitoxin",
    "Alexandrium sxtA",
    "Alexandrium sxtG",
    "saxitoxin biosynthesis genes dinoflagellate",
    "Alexandrium temperature saxitoxin",
    "Alexandrium nitrogen toxin production",
    "Alexandrium salinity toxin profile",
    "Alexandrium transcriptome saxitoxin",
    "saxitoxin evolution dinoflagellates",
    "Centrodinium punctatum saxitoxin",
]

CYANO_QUERIES = [
    "cyanobacteria saxitoxin",
    "cyanobacterial saxitoxin",
    "cyanobacteria paralytic shellfish toxins",
    "Raphidiopsis saxitoxin",
    "Aphanizomenon saxitoxin",
    "Dolichospermum saxitoxin",
    "Cylindrospermopsis saxitoxin",
    "cyanobacteria sxtA",
    "cyanobacteria sxtG",
    "cyanobacterial sxt genes",
    "saxitoxin biosynthesis cyanobacteria",
    "temperature cyanobacteria saxitoxin",
    "nitrogen cyanobacteria saxitoxin",
    "saxitoxin evolution cyanobacteria",
]


# =========================
# PUBMED
# =========================
def pubmed_search(query):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=MAX_PUBMED,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def pubmed_fetch(pmids):
    records = []

    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]

        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="xml",
            retmode="xml"
        )

        data = Entrez.read(handle)
        handle.close()

        for article in data["PubmedArticle"]:
            medline = article["MedlineCitation"]
            art = medline["Article"]

            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", ""))

            abstract = ""
            if "Abstract" in art:
                abstract = " ".join([str(x) for x in art["Abstract"].get("AbstractText", [])])

            journal = str(art.get("Journal", {}).get("Title", ""))

            year = ""
            try:
                year = art["Journal"]["JournalIssue"]["PubDate"].get("Year", "")
            except Exception:
                pass

            doi = ""
            try:
                for eid in art.get("ELocationID", []):
                    if eid.attributes.get("EIdType") == "doi":
                        doi = str(eid)
            except Exception:
                pass

            records.append({
                "source_database": "PubMed",
                "pmid": pmid,
                "doi": doi,
                "year": year,
                "title": title,
                "abstract": abstract,
                "journal": journal,
            })

        time.sleep(0.4)

    return records


# =========================
# OPENALEX
# =========================
def openalex_search(query):
    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per-page": min(MAX_OPENALEX, 200),
        "mailto": Entrez.email
    }

    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()

    data = r.json()
    records = []

    for item in data.get("results", []):
        title = item.get("title") or ""

        abstract = ""
        inv = item.get("abstract_inverted_index")
        if inv:
            words = []
            for word, positions in inv.items():
                for pos in positions:
                    words.append((pos, word))
            abstract = " ".join([w for _, w in sorted(words)])

        doi = item.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        journal = ""
        try:
            journal = item.get("primary_location", {}).get("source", {}).get("display_name") or ""
        except Exception:
            pass

        records.append({
            "source_database": "OpenAlex",
            "pmid": "",
            "doi": doi,
            "year": item.get("publication_year", ""),
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "openalex_id": item.get("id", ""),
        })

    time.sleep(0.5)
    return records


# =========================
# CROSSREF
# =========================
def crossref_search(query):
    url = "https://api.crossref.org/works"

    params = {
        "query": query,
        "rows": MAX_CROSSREF,
        "select": "DOI,title,abstract,published-print,published-online,container-title"
    }

    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()

    data = r.json()
    records = []

    for item in data.get("message", {}).get("items", []):
        title = ""
        if item.get("title"):
            title = item["title"][0]

        abstract = item.get("abstract", "") or ""

        year = ""
        for key in ["published-print", "published-online"]:
            if key in item:
                try:
                    year = item[key]["date-parts"][0][0]
                    break
                except Exception:
                    pass

        journal = ""
        if item.get("container-title"):
            journal = item["container-title"][0]

        records.append({
            "source_database": "CrossRef",
            "pmid": "",
            "doi": item.get("DOI", ""),
            "year": year,
            "title": title,
            "abstract": abstract,
            "journal": journal,
        })

    time.sleep(0.5)
    return records


# =========================
# RUNNERS
# =========================
def run_queries(query_list, corpus_label):
    all_records = []

    for query in query_list:
        print(f"\n=== {corpus_label}: {query} ===")

        # PubMed
        try:
            ids = pubmed_search(query)
            print(f"PubMed found: {len(ids)}")
            recs = pubmed_fetch(ids)
            for r in recs:
                r["query"] = query
                r["corpus"] = corpus_label
            all_records.extend(recs)
        except Exception as e:
            print("PubMed error:", e)

        # OpenAlex
        try:
            recs = openalex_search(query)
            print(f"OpenAlex found: {len(recs)}")
            for r in recs:
                r["query"] = query
                r["corpus"] = corpus_label
            all_records.extend(recs)
        except Exception as e:
            print("OpenAlex error:", e)

        # CrossRef
        try:
            recs = crossref_search(query)
            print(f"CrossRef found: {len(recs)}")
            for r in recs:
                r["query"] = query
                r["corpus"] = corpus_label
            all_records.extend(recs)
        except Exception as e:
            print("CrossRef error:", e)

    df = pd.DataFrame(all_records)

    if df.empty:
        return df

    outfile = OUTDIR / f"{corpus_label}_multidatabase_raw.csv"
    df.to_csv(outfile, index=False, encoding="utf-8-sig")

    print(f"\nSaved raw: {outfile}")
    print("Rows:", len(df))

    return df


def main():
    dino = run_queries(DINO_QUERIES, "dinoflagellate_stx")
    cyano = run_queries(CYANO_QUERIES, "cyanobacteria_stx")

    combined = pd.concat([dino, cyano], ignore_index=True)

    combined.to_csv(
        OUTDIR / "combined_stx_multidatabase_raw.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCombined raw rows:", len(combined))


if __name__ == "__main__":
    main()
