from Bio import Entrez
import pandas as pd
import time

Entrez.email = "your_email@example.com"

QUERY = '(saxitoxin OR "paralytic shellfish toxins" OR PST) AND (cyanobacteria OR cyanobacterial)'

handle = Entrez.esearch(db="pubmed", term=QUERY, retmax=500)
record = Entrez.read(handle)

ids = record["IdList"]

print(f"Found {len(ids)} papers")

papers = []

for i, pmid in enumerate(ids):
    try:
        fetch = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        data = Entrez.read(fetch)

        if not data["PubmedArticle"]:
            continue

        article = data["PubmedArticle"][0]["MedlineCitation"]

        title = article["Article"].get("ArticleTitle", "")
        abstract = ""

        if "Abstract" in article["Article"]:
            abstract = " ".join(article["Article"]["Abstract"]["AbstractText"])

        year = ""

        try:
            year = article["Article"]["Journal"]["JournalIssue"]["PubDate"]["Year"]
        except:
            pass

        papers.append({
            "Collection": "PubMed",
            "Title": title,
            "Abstract": abstract,
            "Journal": "",
            "Year": year,
            "Paper_Id": pmid,
            "Domain": "cyanobacteria",
            "Group": "cyano_train"
        })

        # Be polite to PubMed servers
        time.sleep(0.2)

    except Exception as e:
        print(f"Skipping PMID {pmid}: {e}")
        continue

df = pd.DataFrame(papers)

df.to_csv("cyano_master.csv", index=False, encoding="utf-8-sig")

print(f"Saved cyano_master.csv with {len(df)} papers")
