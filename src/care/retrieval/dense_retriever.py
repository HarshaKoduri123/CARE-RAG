from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from care.embedding.bi_encoder import BiEncoder
from care.utils.text import clean_text, truncate_text


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        device: str,
        normalize_embeddings: bool,
        index_path: str | Path,
        ids_path: str | Path,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"IDs file not found: {self.ids_path}")

        self.encoder = BiEncoder(model_name=model_name, device=device)
        self.index = faiss.read_index(str(self.index_path))

        with self.ids_path.open("r", encoding="utf-8") as f:
            self.ids: List[str] = json.load(f)

        if self.index.ntotal != len(self.ids):
            raise ValueError(
                f"Index/document mismatch: index has {self.index.ntotal} vectors "
                f"but ids file has {len(self.ids)} IDs"
            )

    def encode_query(self, text: str, max_chars: int = 3000) -> np.ndarray:
        text = truncate_text(clean_text(text), max_chars=max_chars)
        emb = self.encoder.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
        )
        return emb.astype(np.float32)

    def search(self, query_text: str, top_k: int = 5, max_chars: int = 3000) -> List[Dict]:
        query_vec = self.encode_query(query_text, max_chars=max_chars)

        scores, indices = self.index.search(query_vec, top_k)

        out: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            out.append(
                {
                    "rank": len(out) + 1,
                    "id": self.ids[idx],
                    "score": float(score),
                    "index": int(idx),
                }
            )
        return out