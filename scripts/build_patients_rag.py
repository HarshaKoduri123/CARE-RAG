#!/usr/bin/env python3


import argparse
import ast
import json
import math
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm




def safe_literal_eval(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return {}
    s = str(value).strip()
    if s == "" or s == "{}":
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def normalize_age(age_obj: Any) -> str:
    parsed = safe_literal_eval(age_obj)
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return "[]"


def normalize_dict_field(d: Any) -> Dict[str, int]:
    parsed = safe_literal_eval(d)
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------
# SQLite schema
# ---------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY,
    patient_uid TEXT UNIQUE,
    source_pmid TEXT,
    source_title TEXT,
    source_file_path TEXT,
    patient_text TEXT,
    age_json TEXT,
    gender TEXT
);

CREATE TABLE IF NOT EXISTS patient_article_labels (
    patient_uid TEXT NOT NULL,
    article_pmid TEXT NOT NULL,
    relevance_score INTEGER NOT NULL,
    PRIMARY KEY (patient_uid, article_pmid)
);

CREATE TABLE IF NOT EXISTS patient_patient_labels (
    patient_uid TEXT NOT NULL,
    similar_patient_uid TEXT NOT NULL,
    similarity_score INTEGER NOT NULL,
    PRIMARY KEY (patient_uid, similar_patient_uid)
);

CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY,
    article_title TEXT,
    abstract TEXT,
    journal TEXT,
    pub_year INTEGER,
    doi TEXT,
    pmcid TEXT,
    authors_json TEXT,
    fetched_ok INTEGER NOT NULL DEFAULT 0,
    raw_xml TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_uid ON patients(patient_uid);
CREATE INDEX IF NOT EXISTS idx_patients_source_pmid ON patients(source_pmid);
CREATE INDEX IF NOT EXISTS idx_pal_article_pmid ON patient_article_labels(article_pmid);
CREATE INDEX IF NOT EXISTS idx_pps_similar_uid ON patient_patient_labels(similar_patient_uid);
"""


# ---------------------------
# PubMed fetch
# ---------------------------

def build_eutils_url(base: str, params: Dict[str, str]) -> str:
    return f"{base}?{urlencode(params)}"


def http_get_text(url: str, user_agent: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_pubmed_xml_batch(
    pmids: List[str],
    email: str,
    api_key: Optional[str],
    tool: str,
    user_agent: str,
    sleep_seconds: float = 0.34,
) -> str:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = build_eutils_url(base, params)
    text = http_get_text(url, user_agent=user_agent)
    time.sleep(sleep_seconds)
    return text


def extract_text_recursive(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return clean_whitespace("".join(elem.itertext()))


def parse_pubmed_article(pubmed_article: ET.Element) -> Dict[str, Any]:
    medline = pubmed_article.find("./MedlineCitation")
    article = medline.find("./Article") if medline is not None else None

    pmid = extract_text_recursive(medline.find("./PMID")) if medline is not None else ""

    article_title = ""
    journal = ""
    pub_year = None
    abstract = ""
    doi = None
    pmcid = None
    authors: List[str] = []

    if article is not None:
        article_title = extract_text_recursive(article.find("./ArticleTitle"))
        journal = extract_text_recursive(article.find("./Journal/Title"))

        year_candidates = [
            article.find("./Journal/JournalIssue/PubDate/Year"),
            article.find("./Journal/JournalIssue/PubDate/MedlineDate"),
            pubmed_article.find("./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year"),
            pubmed_article.find("./PubmedData/History/PubMedPubDate[@PubStatus='entrez']/Year"),
        ]
        for yc in year_candidates:
            txt = extract_text_recursive(yc)
            if txt:
                m = re.search(r"(19|20)\d{2}", txt)
                if m:
                    pub_year = int(m.group(0))
                    break

        abstract_parts = []
        abstract_root = article.find("./Abstract")
        if abstract_root is not None:
            for abs_text in abstract_root.findall("./AbstractText"):
                label = abs_text.attrib.get("Label", "").strip()
                part = extract_text_recursive(abs_text)
                if part:
                    if label:
                        abstract_parts.append(f"{label}: {part}")
                    else:
                        abstract_parts.append(part)
        abstract = "\n".join(abstract_parts).strip()

        author_list = article.find("./AuthorList")
        if author_list is not None:
            for author in author_list.findall("./Author"):
                collective = extract_text_recursive(author.find("./CollectiveName"))
                if collective:
                    authors.append(collective)
                    continue
                last = extract_text_recursive(author.find("./LastName"))
                fore = extract_text_recursive(author.find("./ForeName"))
                name = clean_whitespace(f"{fore} {last}")
                if name:
                    authors.append(name)

    article_ids = pubmed_article.findall("./PubmedData/ArticleIdList/ArticleId")
    for aid in article_ids:
        id_type = aid.attrib.get("IdType", "").lower()
        val = extract_text_recursive(aid)
        if id_type == "doi" and val:
            doi = val
        elif id_type == "pmc" and val:
            pmcid = val

    return {
        "pmid": pmid,
        "article_title": article_title,
        "abstract": abstract,
        "journal": journal,
        "pub_year": pub_year,
        "doi": doi,
        "pmcid": pmcid,
        "authors_json": json.dumps(authors, ensure_ascii=False),
    }


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"Could not parse PubMed XML: {e}")

    out = []
    for node in root.findall("./PubmedArticle"):
        rec = parse_pubmed_article(node)
        if rec.get("pmid"):
            rec["raw_xml"] = None
            rec["fetched_ok"] = 1
            out.append(rec)
    return out




def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF;")
    conn.executescript(SCHEMA_SQL)
    return conn


def insert_patient_rows(
    conn: sqlite3.Connection,
    rows: List[Tuple[int, str, str, str, str, str, str, str]],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO patients
        (patient_id, patient_uid, source_pmid, source_title, source_file_path, patient_text, age_json, gender)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_patient_article_rows(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, str, int]],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO patient_article_labels
        (patient_uid, article_pmid, relevance_score)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def insert_patient_patient_rows(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, str, int]],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO patient_patient_labels
        (patient_uid, similar_patient_uid, similarity_score)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def get_all_unique_article_pmids(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("""
        SELECT DISTINCT article_pmid
        FROM patient_article_labels
        WHERE article_pmid IS NOT NULL AND TRIM(article_pmid) != ''
        ORDER BY article_pmid
    """)
    return [r[0] for r in cur.fetchall()]


def get_missing_article_pmids(conn: sqlite3.Connection, pmids: List[str]) -> List[str]:
    if not pmids:
        return []
    existing = set()
    for batch in chunked(pmids, 999):
        q = ",".join(["?"] * len(batch))
        cur = conn.execute(
            f"SELECT pmid FROM articles WHERE pmid IN ({q}) AND fetched_ok = 1",
            batch,
        )
        existing.update(r[0] for r in cur.fetchall())
    return [p for p in pmids if p not in existing]


def insert_articles(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    to_insert = []
    for r in rows:
        to_insert.append((
            r.get("pmid"),
            r.get("article_title"),
            r.get("abstract"),
            r.get("journal"),
            r.get("pub_year"),
            r.get("doi"),
            r.get("pmcid"),
            r.get("authors_json"),
            int(r.get("fetched_ok", 0)),
            r.get("raw_xml"),
        ))
    conn.executemany(
        """
        INSERT OR REPLACE INTO articles
        (pmid, article_title, abstract, journal, pub_year, doi, pmcid, authors_json, fetched_ok, raw_xml)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        to_insert,
    )


# ---------------------------
# CSV ingestion
# ---------------------------

def ingest_csv_to_sqlite(
    csv_path: str,
    conn: sqlite3.Connection,
    chunksize: int = 5000,
    max_patients: Optional[int] = None,
) -> None:
    usecols = [
        "patient_id",
        "patient_uid",
        "PMID",
        "file_path",
        "title",
        "patient",
        "age",
        "gender",
        "relevant_articles",
        "similar_patients",
    ]

    total_rows = 0
    pbar = tqdm(total=max_patients, desc="Ingesting patients", unit="patient") if max_patients else None

    for chunk_idx, df in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)):
        if max_patients is not None:
            remaining = max_patients - total_rows
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.iloc[:remaining]

        patient_rows = []
        patient_article_rows = []
        patient_patient_rows = []

        for row in df.itertuples(index=False):
            patient_id = int(row.patient_id)
            patient_uid = str(row.patient_uid)
            source_pmid = str(row.PMID) if pd.notna(row.PMID) else ""
            file_path = str(row.file_path) if pd.notna(row.file_path) else ""
            title = str(row.title) if pd.notna(row.title) else ""
            patient_text = str(row.patient) if pd.notna(row.patient) else ""
            age_json = normalize_age(row.age)
            gender = str(row.gender) if pd.notna(row.gender) else ""

            patient_rows.append((
                patient_id,
                patient_uid,
                source_pmid,
                title,
                file_path,
                patient_text,
                age_json,
                gender,
            ))

            rel_articles = normalize_dict_field(row.relevant_articles)
            for article_pmid, relevance_score in rel_articles.items():
                patient_article_rows.append((patient_uid, article_pmid, int(relevance_score)))

            sim_patients = normalize_dict_field(row.similar_patients)
            for similar_uid, similarity_score in sim_patients.items():
                patient_patient_rows.append((patient_uid, similar_uid, int(similarity_score)))

        insert_patient_rows(conn, patient_rows)
        insert_patient_article_rows(conn, patient_article_rows)
        insert_patient_patient_rows(conn, patient_patient_rows)
        conn.commit()

        total_rows += len(df)
        if pbar:
            pbar.update(len(df))

        print(f"[ingest] chunk={chunk_idx} rows={len(df)} total_rows={total_rows}")

    if pbar:
        pbar.close()

    print(f"[ingest] done total_rows={total_rows}")


# ---------------------------
# Fetch abstracts into SQLite
# ---------------------------

def fetch_all_missing_articles(
    conn: sqlite3.Connection,
    email: str,
    api_key: Optional[str],
    tool: str,
    batch_size: int = 200,
    user_agent: Optional[str] = None,
) -> None:
    all_pmids = get_all_unique_article_pmids(conn)
    missing_pmids = get_missing_article_pmids(conn, all_pmids)

    if user_agent is None:
        user_agent = f"{tool}/1.0 ({email})"

    print(f"[fetch] unique labeled article PMIDs = {len(all_pmids)}")
    print(f"[fetch] missing article PMIDs      = {len(missing_pmids)}")

    batches = list(chunked(missing_pmids, batch_size))
    pbar = tqdm(total=len(batches), desc="Fetching article batches", unit="batch")

    for i, batch in enumerate(batches, start=1):
        try:
            xml_text = fetch_pubmed_xml_batch(
                pmids=batch,
                email=email,
                api_key=api_key,
                tool=tool,
                user_agent=user_agent,
                sleep_seconds=0.12 if api_key else 0.40,
            )
            parsed = parse_pubmed_xml(xml_text)

            parsed_pmids = {r["pmid"] for r in parsed}
            missing_in_response = [p for p in batch if p not in parsed_pmids]

            insert_articles(conn, parsed)

            fail_rows = []
            for p in missing_in_response:
                fail_rows.append({
                    "pmid": p,
                    "article_title": None,
                    "abstract": None,
                    "journal": None,
                    "pub_year": None,
                    "doi": None,
                    "pmcid": None,
                    "authors_json": "[]",
                    "fetched_ok": 0,
                    "raw_xml": None,
                })
            if fail_rows:
                insert_articles(conn, fail_rows)

            conn.commit()
            pbar.update(1)
            pbar.set_postfix(parsed=len(parsed), unresolved=len(missing_in_response))

        except Exception as e:
            print(f"[fetch] batch={i} failed: {e}", file=sys.stderr)
            time.sleep(2.0)
            pbar.update(1)

    pbar.close()


# ---------------------------
# JSONL exports
# ---------------------------

def export_jsonl(conn: sqlite3.Connection, outdir: str, max_articles_per_patient: int = 50) -> None:
    ensure_dir(outdir)

    patients_path = os.path.join(outdir, "patients.jsonl")
    articles_path = os.path.join(outdir, "articles.jsonl")
    rag_path = os.path.join(outdir, "rag_examples.jsonl")

    with open(patients_path, "w", encoding="utf-8") as f:
        cur = conn.execute("""
            SELECT patient_id, patient_uid, source_pmid, source_title, source_file_path,
                   patient_text, age_json, gender
            FROM patients
            ORDER BY patient_id
        """)
        cols = [d[0] for d in cur.description]
        for row in cur:
            obj = dict(zip(cols, row))
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(articles_path, "w", encoding="utf-8") as f:
        cur = conn.execute("""
            SELECT pmid, article_title, abstract, journal, pub_year, doi, pmcid, authors_json, fetched_ok
            FROM articles
            ORDER BY pmid
        """)
        cols = [d[0] for d in cur.description]
        for row in cur:
            obj = dict(zip(cols, row))
            try:
                obj["authors"] = json.loads(obj.pop("authors_json") or "[]")
            except Exception:
                obj["authors"] = []
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(rag_path, "w", encoding="utf-8") as f:
        cur = conn.execute("""
            SELECT patient_id, patient_uid, source_pmid, source_title, source_file_path,
                   patient_text, age_json, gender
            FROM patients
            ORDER BY patient_id
        """)
        for row in cur:
            patient_id, patient_uid, source_pmid, source_title, source_file_path, patient_text, age_json, gender = row

            article_cur = conn.execute("""
                SELECT pal.article_pmid, pal.relevance_score,
                       a.article_title, a.abstract, a.journal, a.pub_year, a.doi, a.pmcid, a.fetched_ok
                FROM patient_article_labels pal
                LEFT JOIN articles a ON a.pmid = pal.article_pmid
                WHERE pal.patient_uid = ?
                ORDER BY pal.relevance_score DESC, pal.article_pmid
                LIMIT ?
            """, (patient_uid, max_articles_per_patient))
            relevant_articles = []
            for ar in article_cur.fetchall():
                relevant_articles.append({
                    "pmid": ar[0],
                    "relevance_score": ar[1],
                    "title": ar[2],
                    "abstract": ar[3],
                    "journal": ar[4],
                    "pub_year": ar[5],
                    "doi": ar[6],
                    "pmcid": ar[7],
                    "fetched_ok": ar[8],
                })

            sim_cur = conn.execute("""
                SELECT similar_patient_uid, similarity_score
                FROM patient_patient_labels
                WHERE patient_uid = ?
                ORDER BY similarity_score DESC, similar_patient_uid
            """, (patient_uid,))
            similar_patients = [
                {"similar_patient_uid": s[0], "similarity_score": s[1]}
                for s in sim_cur.fetchall()
            ]

            obj = {
                "patient_id": patient_id,
                "patient_uid": patient_uid,
                "source_pmid": source_pmid,
                "source_title": source_title,
                "source_file_path": source_file_path,
                "patient_text": patient_text,
                "age": json.loads(age_json) if age_json else [],
                "gender": gender,
                "relevant_articles": relevant_articles,
                "similar_patients": similar_patients,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[export] wrote {patients_path}")
    print(f"[export] wrote {articles_path}")
    print(f"[export] wrote {rag_path}")


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to PMC-Patients.csv")
    p.add_argument("--db", default="pmc_patients_rag.db", help="SQLite database output path")
    p.add_argument("--outdir", default="rag_output", help="Output directory for JSONL files")
    p.add_argument("--email", required=True, help="Email for NCBI E-utilities")
    p.add_argument("--api-key", default=None, help="Optional NCBI API key")
    p.add_argument("--tool", default="pmc_patients_rag_builder", help="Tool name sent to NCBI")
    p.add_argument("--ingest-chunksize", type=int, default=5000, help="CSV ingest chunk size")
    p.add_argument("--fetch-batch-size", type=int, default=200, help="PMIDs per efetch batch")
    p.add_argument("--max-articles-per-patient", type=int, default=50, help="Max linked articles in rag_examples.jsonl")
    p.add_argument("--max-patients", type=int, default=None, help="Only ingest first N patients")
    p.add_argument("--skip-ingest", action="store_true", help="Skip CSV ingestion")
    p.add_argument("--skip-fetch", action="store_true", help="Skip article fetching")
    p.add_argument("--skip-export", action="store_true", help="Skip JSONL export")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(os.path.dirname(args.db) or ".")
    ensure_dir(args.outdir)

    conn = connect_db(args.db)

    if not args.skip_ingest:
        print("[main] ingesting CSV into SQLite")
        ingest_csv_to_sqlite(
            args.csv,
            conn,
            chunksize=args.ingest_chunksize,
            max_patients=args.max_patients,
        )

    if not args.skip_fetch:
        print("[main] fetching PubMed abstracts/metadata")
        fetch_all_missing_articles(
            conn=conn,
            email=args.email,
            api_key=args.api_key,
            tool=args.tool,
            batch_size=args.fetch_batch_size,
        )

    if not args.skip_export:
        print("[main] exporting JSONL")
        export_jsonl(conn, args.outdir, max_articles_per_patient=args.max_articles_per_patient)

    conn.close()
    print("done")


if __name__ == "__main__":
    main()