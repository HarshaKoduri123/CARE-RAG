from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        required=True,
        choices=["patients", "articles"],
        help="Which embedding corpus to index",
    )
    p.add_argument(
        "--embeddings-dir",
        default="outputs/embeddings",
        help="Directory containing saved .npy embedding files",
    )
    p.add_argument(
        "--index-dir",
        default="data/indexes",
        help="Directory where FAISS index will be saved",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus == "patients":
        emb_path = embeddings_dir / "patient_embeddings.npy"
        index_path = index_dir / "patient_faiss.index"
    else:
        emb_path = embeddings_dir / "article_embeddings.npy"
        index_path = index_dir / "article_faiss.index"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    embeddings = np.load(emb_path)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    print(f"Saved FAISS index to: {index_path}")
    print(f"Indexed vectors:      {index.ntotal:,}")
    print(f"Embedding dim:        {dim}")


if __name__ == "__main__":
    main()