from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from care.data.loaders import iter_articles_jsonl
from care.retrieval.dense_retriever import DenseRetriever


class PatientArticleRetriever:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg

        model_name = cfg["model"]["name"]
        device = cfg["model"]["device"]
        normalize = cfg["model"].get("normalize", True)

        embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
        index_dir = Path(cfg["paths"]["index_dir"])

        corpus_cfg = cfg["corpora"]["articles"]

        index_path = index_dir / corpus_cfg["index_file"]
        ids_path = embeddings_dir / corpus_cfg["ids_file"]

        self.retriever = DenseRetriever(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize,
            index_path=index_path,
            ids_path=ids_path,
        )

        self.article_store = self._load_article_store()

    def _load_article_store(self) -> Dict[str, Dict]:
        store: Dict[str, Dict] = {}
        for row in iter_articles_jsonl():
            pmid = str(row["pmid"])
            store[pmid] = row
        return store

    def retrieve(self, patient_text: str, top_k: int, max_query_chars: int) -> List[Dict]:
        results = self.retriever.search(
            query_text=patient_text,
            top_k=top_k,
            max_chars=max_query_chars,
        )

        enriched: List[Dict] = []
        for item in results:
            pmid = item["id"]
            article = self.article_store.get(pmid, {})

            enriched.append(
                {
                    "rank": item["rank"],
                    "pmid": pmid,
                    "score": item["score"],
                    "article_title": article.get("article_title", ""),
                    "abstract": article.get("abstract", ""),
                    "journal": article.get("journal", ""),
                    "pub_year": article.get("pub_year", None),
                    "doi": article.get("doi", None),
                    "pmcid": article.get("pmcid", None),
                }
            )

        return enriched