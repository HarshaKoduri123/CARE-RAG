from __future__ import annotations

from typing import List, Dict


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def retrieve(
        self,
        query_text: str,
        dense_top_k: int,
        bm25_top_k: int,
    ) -> List[Dict]:

        dense_results = self.dense.search(query_text, top_k=dense_top_k)
        bm25_results = self.bm25.search(query_text, top_k=bm25_top_k)

        merged = {}

        # add dense
        for item in dense_results:
            merged[item["id"]] = {
                "id": item["id"],
                "dense_score": item["score"],
                "bm25_score": 0.0,
            }

        # add bm25
        for item in bm25_results:
            if item["id"] not in merged:
                merged[item["id"]] = {
                    "id": item["id"],
                    "dense_score": 0.0,
                    "bm25_score": item["bm25_score"],
                }
            else:
                merged[item["id"]]["bm25_score"] = item["bm25_score"]

        # convert to list
        results = list(merged.values())

        return results