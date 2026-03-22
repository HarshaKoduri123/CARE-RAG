from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from care.data.loaders import iter_articles_jsonl, iter_patients_jsonl
from care.embedding.bi_encoder import BiEncoder
from care.utils.io import ensure_parent_dir
from care.utils.text import clean_text, truncate_text
from care.utils.config import load_config


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to embedding YAML config")
    return p.parse_args()


def build_patient_text(row: Dict, max_chars: int) -> str:
    text = clean_text(row.get("patient_text", ""))
    return truncate_text(text, max_chars=max_chars)


def build_article_text(row: Dict, max_chars: int) -> str:
    title = clean_text(row.get("article_title", ""))
    abstract = clean_text(row.get("abstract", ""))

    if title and abstract:
        text = f"{title}. {abstract}"
    elif title:
        text = title
    else:
        text = abstract

    return truncate_text(text, max_chars=max_chars)


def collect_patients(limit: int | None, max_chars: int) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []

    for row in tqdm(iter_patients_jsonl(limit=limit), desc="Loading patients"):
        patient_uid = str(row["patient_uid"])
        text = build_patient_text(row, max_chars=max_chars)
        if not text:
            continue
        ids.append(patient_uid)
        texts.append(text)

    return ids, texts


def collect_articles(limit: int | None, max_chars: int) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []

    for row in tqdm(iter_articles_jsonl(limit=limit), desc="Loading articles"):
        pmid = str(row["pmid"])
        text = build_article_text(row, max_chars=max_chars)
        if not text:
            continue
        ids.append(pmid)
        texts.append(text)

    return ids, texts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg["model"]["name"]
    device = cfg["model"]["device"]
    normalize = cfg["model"].get("normalize", True)

    corpus = cfg["data"]["corpus"]
    max_chars = cfg["data"]["max_chars"]
    limit = cfg["data"]["limit"]

    batch_size = cfg["runtime"]["batch_size"]

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if corpus == "patients":
        ids, texts = collect_patients(limit=limit, max_chars=max_chars)
        emb_path = output_dir / "patient_embeddings.npy"
        ids_path = output_dir / "patient_ids.json"
        meta_path = output_dir / "patient_meta.json"
    else:
        ids, texts = collect_articles(limit=limit, max_chars=max_chars)
        emb_path = output_dir / "article_embeddings.npy"
        ids_path = output_dir / "article_ids.json"
        meta_path = output_dir / "article_meta.json"

    print(f"Loaded {len(ids):,} {corpus} records for embedding")

    encoder = BiEncoder(model_name=model_name, device=device)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )

    ensure_parent_dir(emb_path)
    np.save(emb_path, embeddings)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    meta = {
        "corpus": corpus,
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "limit": limit,
        "max_chars": max_chars,
        "normalize_embeddings": normalize,
        "num_records": len(ids),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else None,
        "embeddings_file": str(emb_path),
        "ids_file": str(ids_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved IDs to:        {ids_path}")
    print(f"Saved metadata to:   {meta_path}")
    print(f"Embedding shape:     {embeddings.shape}")


if __name__ == "__main__":
    main()