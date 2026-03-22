from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from care.data.loaders import iter_articles_jsonl, iter_patients_jsonl
from care.retrieval.bm25_index import BM25Index
from care.utils.text import clean_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        required=True,
        choices=["patients", "articles"],
        help="Which corpus to build BM25 index for",
    )
    return p.parse_args()


def build_article_text(row: dict) -> str:
    title = clean_text(row.get("article_title", ""))
    abstract = clean_text(row.get("abstract", ""))

    if title and abstract:
        return f"{title}. {abstract}"
    elif title:
        return title
    else:
        return abstract


def build_patient_text(row: dict) -> str:
    return clean_text(row.get("patient_text", ""))


def main() -> None:
    args = parse_args()

    texts = []
    ids = []

    if args.corpus == "patients":
        for row in iter_patients_jsonl():
            patient_uid = str(row["patient_uid"])
            text = build_patient_text(row)

            if not text:
                continue

            ids.append(patient_uid)
            texts.append(text)

        out_path = Path("data/indexes/patient_bm25.pkl")

    else:
        for row in iter_articles_jsonl():
            pmid = str(row["pmid"])
            text = build_article_text(row)

            if not text:
                continue

            ids.append(pmid)
            texts.append(text)

        out_path = Path("data/indexes/article_bm25.pkl")

    print(f"Loaded {len(ids):,} {args.corpus}")

    bm25 = BM25Index(texts=texts, ids=ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved BM25 index: {out_path}")


if __name__ == "__main__":
    main()