from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.care.database.connection import get_db_connection
from src.care.database.queries import get_patient_by_uid, iter_patient_patient_labels
from src.care.utils.io import ensure_parent_dir
from src.care.utils.text import clean_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/interim/pmc_patients_rag_full.db")
    p.add_argument("--out", default="data/processed/patient_patient_pairs.jsonl")
    p.add_argument("--min-similarity", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_db_connection(args.db)

    out_path = Path(args.out)
    ensure_parent_dir(out_path)

    total = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for label in tqdm(
            iter_patient_patient_labels(conn, min_similarity=args.min_similarity, limit=args.limit),
            desc="Building patient-patient pairs",
        ):
            p1 = get_patient_by_uid(conn, label["patient_uid"])
            p2 = get_patient_by_uid(conn, label["similar_patient_uid"])

            if p1 is None or p2 is None:
                skipped += 1
                continue

            row = {
                "patient_uid": p1["patient_uid"],
                "patient_text": clean_text(p1["patient_text"]),
                "similar_patient_uid": p2["patient_uid"],
                "similar_patient_text": clean_text(p2["patient_text"]),
                "similarity_score": int(label["similarity_score"]),
            }
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    conn.close()
    print(f"done | wrote={total} | skipped={skipped} | output={out_path}")


if __name__ == "__main__":
    main()