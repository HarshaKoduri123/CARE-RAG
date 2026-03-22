from __future__ import annotations

import re
from typing import Dict, List

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, texts: List[str], ids: List[str]):
        self.ids = ids
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {"id": doc_id, "bm25_score": float(score)}
            for doc_id, score in ranked
        ]