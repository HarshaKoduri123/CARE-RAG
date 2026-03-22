from __future__ import annotations

from typing import Dict, List, Tuple

from care.reranking.cross_encoder import CrossEncoderReranker
from care.utils.text import clean_text, truncate_text


class PatientPatientReranker:
    def __init__(self, cfg: Dict) -> None:
        model_name = cfg["model"]["name"]
        device = cfg["model"]["device"]
        self.batch_size = cfg["runtime"]["batch_size"]
        self.max_query_chars = cfg["search"]["max_query_chars"]
        self.max_doc_chars = cfg["search"]["max_doc_chars"]

        self.reranker = CrossEncoderReranker(
            model_name=model_name,
            device=device,
        )

    def _build_patient_text(self, item: Dict) -> str:
        text = clean_text(item.get("patient_text", ""))
        return truncate_text(text, max_chars=self.max_doc_chars)

    def rerank(self, patient_text: str, similar_patients: List[Dict], top_k: int) -> List[Dict]:
        if not similar_patients:
            return []

        query = truncate_text(clean_text(patient_text), max_chars=self.max_query_chars)

        pairs: List[Tuple[str, str]] = []
        for item in similar_patients:
            doc_text = self._build_patient_text(item)
            pairs.append((query, doc_text))

        scores = self.reranker.score_pairs(
            pairs,
            batch_size=self.batch_size,
        )

        reranked: List[Dict] = []
        for item, score in zip(similar_patients, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        final = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            item["rerank_rank"] = rank
            final.append(item)

        return final