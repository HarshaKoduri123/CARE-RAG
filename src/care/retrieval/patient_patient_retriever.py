from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from care.data.loaders import iter_patients_jsonl
from care.retrieval.dense_retriever import DenseRetriever


class PatientPatientRetriever:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg

        model_name = cfg["model"]["name"]
        device = cfg["model"]["device"]
        normalize = cfg["model"].get("normalize", True)

        embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
        index_dir = Path(cfg["paths"]["index_dir"])

        corpus_cfg = cfg["corpora"]["patients"]

        index_path = index_dir / corpus_cfg["index_file"]
        ids_path = embeddings_dir / corpus_cfg["ids_file"]

        self.retriever = DenseRetriever(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize,
            index_path=index_path,
            ids_path=ids_path,
        )

        self.patient_store = self._load_patient_store()

    def _load_patient_store(self) -> Dict[str, Dict]:
        store: Dict[str, Dict] = {}
        for row in iter_patients_jsonl():
            patient_uid = str(row["patient_uid"])
            store[patient_uid] = row
        return store

    def retrieve(
        self,
        patient_text: str,
        top_k: int,
        max_query_chars: int,
        exclude_patient_uid: Optional[str] = None,
        exclude_source_pmid: Optional[str] = None,
    ) -> List[Dict]:
        # pull a few extra in case self-match / same-source items need filtering
        search_top_k = top_k + 10

        results = self.retriever.search(
            query_text=patient_text,
            top_k=search_top_k,
            max_chars=max_query_chars,
        )

        enriched: List[Dict] = []
        for item in results:
            patient_uid = str(item["id"])

            if exclude_patient_uid is not None and patient_uid == str(exclude_patient_uid):
                continue

            patient = self.patient_store.get(patient_uid, {})
            source_pmid = str(patient.get("source_pmid", ""))

            if exclude_source_pmid is not None and source_pmid == str(exclude_source_pmid):
                continue

            enriched.append(
                {
                    "rank": len(enriched) + 1,
                    "patient_uid": patient_uid,
                    "score": item["score"],
                    "patient_text": patient.get("patient_text", ""),
                    "source_pmid": source_pmid,
                    "source_title": patient.get("source_title", ""),
                    "gender": patient.get("gender", ""),
                    "age": patient.get("age_json", patient.get("age", "")),
                }
            )

            if len(enriched) >= top_k:
                break

        return enriched