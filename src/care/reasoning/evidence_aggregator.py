from __future__ import annotations

from typing import Dict, List

from care.utils.text import clean_text, truncate_text


class EvidenceAggregator:
    def __init__(self, cfg: Dict) -> None:
        reasoning_cfg = cfg["reasoning"]
        self.max_article_chars = reasoning_cfg["max_article_chars"]
        self.max_patient_chars = reasoning_cfg["max_patient_chars"]
        self.max_query_chars = reasoning_cfg["max_query_chars"]
        self.max_articles = reasoning_cfg["max_articles"]
        self.max_similar_patients = reasoning_cfg["max_similar_patients"]

    def _normalize_query(self, query_text: str) -> str:
        return truncate_text(clean_text(query_text), max_chars=self.max_query_chars)

    def _aggregate_articles(self, articles: List[Dict]) -> List[Dict]:
        out: List[Dict] = []

        for item in articles[: self.max_articles]:
            title = clean_text(item.get("article_title", ""))
            abstract = truncate_text(
                clean_text(item.get("abstract", "")),
                max_chars=self.max_article_chars,
            )

            out.append(
                {
                    "rank": item.get("rerank_rank", item.get("rank")),
                    "pmid": item.get("pmid"),
                    "dense_score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                    "title": title,
                    "abstract": abstract,
                    "journal": clean_text(item.get("journal", "")),
                    "pub_year": item.get("pub_year"),
                    "doi": item.get("doi"),
                    "pmcid": item.get("pmcid"),
                }
            )

        return out

    def _aggregate_patients(self, patients: List[Dict]) -> List[Dict]:
        out: List[Dict] = []

        for item in patients[: self.max_similar_patients]:
            patient_text = truncate_text(
                clean_text(item.get("patient_text", "")),
                max_chars=self.max_patient_chars,
            )

            out.append(
                {
                    "rank": item.get("rerank_rank", item.get("rank")),
                    "patient_uid": item.get("patient_uid"),
                    "dense_score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                    "source_pmid": item.get("source_pmid"),
                    "source_title": clean_text(item.get("source_title", "")),
                    "gender": clean_text(item.get("gender", "")),
                    "age": item.get("age"),
                    "patient_text": patient_text,
                }
            )

        return out

    def build(
        self,
        query_text: str,
        reranked_articles: List[Dict],
        reranked_patients: List[Dict],
    ) -> Dict:
        return {
            "query_text": self._normalize_query(query_text),
            "articles": self._aggregate_articles(reranked_articles),
            "similar_patients": self._aggregate_patients(reranked_patients),
        }