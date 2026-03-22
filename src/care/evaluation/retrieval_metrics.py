from __future__ import annotations

import math
from typing import Dict, List, Set


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for x in retrieved_k if x in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    for rank, item_id in enumerate(retrieved_k, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    retrieved_k = retrieved_ids[:k]

    dcg = 0.0
    for rank, item_id in enumerate(retrieved_k, start=1):
        rel = relevance_map.get(item_id, 0.0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg