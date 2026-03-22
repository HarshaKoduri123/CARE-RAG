from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List


class BM25Retriever:
    def __init__(self, index_path: Path):
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

    def search(self, query: str, top_k: int) -> List[Dict]:
        return self.index.search(query, top_k=top_k)