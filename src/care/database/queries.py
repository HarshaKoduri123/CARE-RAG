from __future__ import annotations

import sqlite3
from typing import Dict, Generator, List, Optional


def iter_patients(conn: sqlite3.Connection, limit: Optional[int] = None) -> Generator[dict, None, None]:
    sql = """
        SELECT patient_id, patient_uid, source_pmid, source_title, source_file_path,
               patient_text, age_json, gender
        FROM patients
        ORDER BY patient_id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.execute(sql)
    for row in cur:
        yield dict(row)


def iter_articles(conn: sqlite3.Connection, limit: Optional[int] = None) -> Generator[dict, None, None]:
    sql = """
        SELECT pmid, article_title, abstract, journal, pub_year, doi, pmcid,
               authors_json, fetched_ok
        FROM articles
        ORDER BY pmid
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.execute(sql)
    for row in cur:
        yield dict(row)


def iter_patient_article_labels(
    conn: sqlite3.Connection,
    min_relevance: int = 1,
    limit: Optional[int] = None,
) -> Generator[dict, None, None]:
    sql = """
        SELECT patient_uid, article_pmid, relevance_score
        FROM patient_article_labels
        WHERE relevance_score >= ?
        ORDER BY patient_uid, relevance_score DESC, article_pmid
    """
    params = [min_relevance]
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.execute(sql, params)
    for row in cur:
        yield dict(row)


def iter_patient_patient_labels(
    conn: sqlite3.Connection,
    min_similarity: int = 1,
    limit: Optional[int] = None,
) -> Generator[dict, None, None]:
    sql = """
        SELECT patient_uid, similar_patient_uid, similarity_score
        FROM patient_patient_labels
        WHERE similarity_score >= ?
        ORDER BY patient_uid, similarity_score DESC, similar_patient_uid
    """
    params = [min_similarity]
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.execute(sql, params)
    for row in cur:
        yield dict(row)


def get_patient_by_uid(conn: sqlite3.Connection, patient_uid: str) -> Optional[dict]:
    cur = conn.execute(
        """
        SELECT patient_id, patient_uid, source_pmid, source_title, source_file_path,
               patient_text, age_json, gender
        FROM patients
        WHERE patient_uid = ?
        """,
        (patient_uid,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_article_by_pmid(conn: sqlite3.Connection, pmid: str) -> Optional[dict]:
    cur = conn.execute(
        """
        SELECT pmid, article_title, abstract, journal, pub_year, doi, pmcid,
               authors_json, fetched_ok
        FROM articles
        WHERE pmid = ?
        """,
        (pmid,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_db_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    tables = [
        "patients",
        "articles",
        "patient_article_labels",
        "patient_patient_labels",
    ]
    out: Dict[str, int] = {}
    for table in tables:
        cur = conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
        out[table] = int(cur.fetchone()["n"])
    return out


def get_relevant_articles_for_patient(conn, patient_uid: str) -> dict:
    cur = conn.execute(
        """
        SELECT article_pmid, relevance_score
        FROM patient_article_labels
        WHERE patient_uid = ?
        """,
        (patient_uid,),
    )
    return {str(row[0]): float(row[1]) for row in cur.fetchall()}


def get_similar_patients_for_patient(conn, patient_uid: str) -> dict:
    cur = conn.execute(
        """
        SELECT similar_patient_uid, similarity_score
        FROM patient_patient_labels
        WHERE patient_uid = ?
        """,
        (patient_uid,),
    )
    return {str(row[0]): float(row[1]) for row in cur.fetchall()}