from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from care.data.loaders import iter_patients_jsonl
from care.database.connection import get_db_connection
from care.database.queries import (
    get_relevant_articles_for_patient,
    get_similar_patients_for_patient,
)
from care.evaluation.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k
from care.llm.generation import LLMGenerator
from care.llm.ollama_client import OllamaClient
from care.reasoning.evidence_aggregator import EvidenceAggregator
from care.reasoning.llm_query_builder import LLMClinicalQueryBuilder
from care.reasoning.output_formatter import ReasoningOutputFormatter
from care.reasoning.output_parser import ReasoningOutputParser
from care.reasoning.prompt_builder import PromptBuilder
from care.reasoning.query_builder import ClinicalQueryBuilder
from care.retrieval.bm25_retriever import BM25Retriever
from care.retrieval.hybrid_retriever import HybridRetriever
from care.retrieval.patient_article_retriever import PatientArticleRetriever
from care.retrieval.patient_patient_retriever import PatientPatientRetriever
from care.reranking.patient_article_reranker import PatientArticleReranker
from care.reranking.patient_patient_reranker import PatientPatientReranker
from care.utils.config import load_config
from care.utils.text import truncate_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    p.add_argument("--reranker-config", default="configs/reranker.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--patient-text", default=None)
    p.add_argument("--patient-uid", default=None)
    p.add_argument("--retrieval-query", default=None, help="Manual retrieval query override")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument(
        "--save-report",
        default="outputs/predictions/run_report.txt",
        help="Path to save the full text report",
    )
    return p.parse_args()


def get_patient_record_by_uid(patient_uid: str) -> dict:
    for row in iter_patients_jsonl():
        if str(row["patient_uid"]) == str(patient_uid):
            return row
    raise ValueError(f"patient_uid not found in patients.jsonl: {patient_uid}")


def build_retrieval_query(args: argparse.Namespace, llm_cfg: dict, query_text: str) -> str:
    if args.retrieval_query:
        return args.retrieval_query

    query_mode = llm_cfg.get("query_builder", {}).get("mode", "llm")

    if query_mode == "llm":
        llm_builder_cfg = llm_cfg.get("llm_query_builder", {})
        provider = llm_builder_cfg.get("provider", "ollama")

        if provider != "ollama":
            raise ValueError(f"Unsupported llm_query_builder provider: {provider}")

        ollama_client = OllamaClient(
            base_url=llm_builder_cfg.get("base_url", "http://localhost:11434"),
            model=llm_builder_cfg.get("model", "llama3.1:8b"),
            timeout_seconds=llm_builder_cfg.get("timeout_seconds", 120),
        )

        def ollama_generate_fn(prompt: str) -> str:
            return ollama_client.generate(
                prompt=prompt,
                temperature=llm_builder_cfg.get("temperature", 0.0),
                max_tokens=llm_builder_cfg.get("max_tokens", 200),
            )

        llm_query_builder = LLMClinicalQueryBuilder(
            cfg=llm_cfg,
            llm_generate_fn=ollama_generate_fn,
        )
        return llm_query_builder.build(query_text)

    query_builder = ClinicalQueryBuilder(llm_cfg)
    return query_builder.build(query_text)


def main() -> None:
    args = parse_args()

    retrieval_cfg = load_config(args.retrieval_config)
    reranker_cfg = load_config(args.reranker_config)
    llm_cfg = load_config(args.llm_config)

    retrieve_top_k = reranker_cfg["search"]["retrieve_top_k"]
    rerank_top_k = args.top_k or reranker_cfg["search"]["rerank_top_k"]

    dense_top_k = retrieval_cfg.get("retrieval", {}).get("dense_top_k", 50)
    bm25_top_k = retrieval_cfg.get("retrieval", {}).get("bm25_top_k", 50)

    exclude_patient_uid: Optional[str] = None
    exclude_source_pmid: Optional[str] = None

    if args.patient_text:
        query_text = args.patient_text
        query_source = "raw_text"
    elif args.patient_uid:
        patient_row = get_patient_record_by_uid(args.patient_uid)
        query_text = patient_row["patient_text"]
        query_source = f"patient_uid:{args.patient_uid}"
        exclude_patient_uid = str(patient_row["patient_uid"])
        exclude_source_pmid = str(patient_row.get("source_pmid", "")) or None
    else:
        raise ValueError("Provide either --patient-text or --patient-uid")

    retrieval_query = build_retrieval_query(args, llm_cfg, query_text)

    article_retriever = PatientArticleRetriever(retrieval_cfg)
    patient_retriever = PatientPatientRetriever(retrieval_cfg)

    article_bm25 = BM25Retriever(Path("data/indexes/article_bm25.pkl"))
    patient_bm25 = BM25Retriever(Path("data/indexes/patient_bm25.pkl"))

    hybrid_article_retriever = HybridRetriever(
        dense_retriever=article_retriever.retriever,
        bm25_retriever=article_bm25,
    )
    hybrid_patient_retriever = HybridRetriever(
        dense_retriever=patient_retriever.retriever,
        bm25_retriever=patient_bm25,
    )

    article_reranker = PatientArticleReranker(reranker_cfg)
    patient_reranker = PatientPatientReranker(reranker_cfg)

    aggregator = EvidenceAggregator(llm_cfg)
    prompt_builder = PromptBuilder(llm_cfg)

    # -------------------------
    # ARTICLE HYBRID RETRIEVAL
    # -------------------------
    hybrid_article_hits = hybrid_article_retriever.retrieve(
        query_text=retrieval_query,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
    )

    retrieved_articles = []
    seen_pmids = set()

    for item in hybrid_article_hits:
        pmid = str(item["id"])
        if pmid in seen_pmids:
            continue

        article = article_retriever.article_store.get(pmid, {})
        if not article:
            continue

        retrieved_articles.append(
            {
                "rank": len(retrieved_articles) + 1,
                "pmid": pmid,
                "score": float(item.get("dense_score", 0.0)),
                "bm25_score": float(item.get("bm25_score", 0.0)),
                "article_title": article.get("article_title", ""),
                "abstract": article.get("abstract", ""),
                "journal": article.get("journal", ""),
                "pub_year": article.get("pub_year", None),
                "doi": article.get("doi", None),
                "pmcid": article.get("pmcid", None),
            }
        )
        seen_pmids.add(pmid)

        if len(retrieved_articles) >= retrieve_top_k:
            break

    # -------------------------
    # PATIENT HYBRID RETRIEVAL
    # -------------------------
    hybrid_patient_hits = hybrid_patient_retriever.retrieve(
        query_text=retrieval_query,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
    )

    retrieved_patients = []
    seen_patient_uids = set()

    for item in hybrid_patient_hits:
        patient_uid = str(item["id"])

        if exclude_patient_uid is not None and patient_uid == exclude_patient_uid:
            continue

        patient = patient_retriever.patient_store.get(patient_uid, {})
        if not patient:
            continue

        source_pmid = str(patient.get("source_pmid", ""))

        if exclude_source_pmid is not None and source_pmid == exclude_source_pmid:
            continue

        if patient_uid in seen_patient_uids:
            continue

        retrieved_patients.append(
            {
                "rank": len(retrieved_patients) + 1,
                "patient_uid": patient_uid,
                "score": float(item.get("dense_score", 0.0)),
                "bm25_score": float(item.get("bm25_score", 0.0)),
                "patient_text": patient.get("patient_text", ""),
                "source_pmid": source_pmid,
                "source_title": patient.get("source_title", ""),
                "gender": patient.get("gender", ""),
                "age": patient.get("age_json", patient.get("age", "")),
            }
        )
        seen_patient_uids.add(patient_uid)

        if len(retrieved_patients) >= retrieve_top_k:
            break

    # -------------------------
    # RERANK
    # -------------------------
    reranked_articles = article_reranker.rerank(
        patient_text=retrieval_query,
        retrieved_articles=retrieved_articles,
        top_k=rerank_top_k,
    )

    reranked_patients = patient_reranker.rerank(
        patient_text=retrieval_query,
        similar_patients=retrieved_patients,
        top_k=rerank_top_k,
    )

    # -------------------------
    # METRICS (only for patient_uid queries with ground truth)
    # -------------------------
    article_metrics = None
    patient_metrics = None

    if args.patient_uid:
        conn = get_db_connection()

        article_relevance_map = get_relevant_articles_for_patient(conn, args.patient_uid)
        patient_relevance_map = get_similar_patients_for_patient(conn, args.patient_uid)

        conn.close()

        reranked_article_ids = [str(x["pmid"]) for x in reranked_articles]
        reranked_patient_ids = [str(x["patient_uid"]) for x in reranked_patients]

        article_metrics = {
            "Recall@5": recall_at_k(reranked_article_ids, set(article_relevance_map.keys()), 5),
            "MRR@5": mrr_at_k(reranked_article_ids, set(article_relevance_map.keys()), 5),
            "nDCG@5": ndcg_at_k(reranked_article_ids, article_relevance_map, 5),
        }

        patient_metrics = {
            "Recall@5": recall_at_k(reranked_patient_ids, set(patient_relevance_map.keys()), 5),
            "MRR@5": mrr_at_k(reranked_patient_ids, set(patient_relevance_map.keys()), 5),
            "nDCG@5": ndcg_at_k(reranked_patient_ids, patient_relevance_map, 5),
        }

    # -------------------------
    # EVIDENCE + PROMPT + REASONING
    # -------------------------
    evidence_pack = aggregator.build(
        query_text=query_text,
        reranked_articles=reranked_articles,
        reranked_patients=reranked_patients,
    )

    reasoning_prompt = prompt_builder.build_reasoning_prompt(evidence_pack)

    generator = LLMGenerator(llm_cfg)
    parser = ReasoningOutputParser()
    formatter = ReasoningOutputFormatter()

    reasoning_output_text = generator.generate(reasoning_prompt)
    reasoning_output = parser.parse(reasoning_output_text)
    formatted_reasoning_output = formatter.format_for_console(reasoning_output)

    # -------------------------
    # CONSOLE OUTPUT
    # -------------------------
    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(truncate_text(query_text, max_chars=600))

    print("\n" + "=" * 80)
    print("RETRIEVAL QUERY")
    print("=" * 80)
    print(retrieval_query)

    print("\n" + "=" * 80)
    print("EVIDENCE PACK SUMMARY")
    print("=" * 80)
    print(f"Articles: {len(evidence_pack['articles'])}")
    print(f"Similar patients: {len(evidence_pack['similar_patients'])}")
    if exclude_patient_uid:
        print(f"Excluded patient_uid: {exclude_patient_uid}")
    if exclude_source_pmid:
        print(f"Excluded source_pmid: {exclude_source_pmid}")

    if article_metrics is not None:
        print("\n" + "=" * 80)
        print("ARTICLE RETRIEVAL METRICS")
        print("=" * 80)
        for k, v in article_metrics.items():
            print(f"{k}: {v:.4f}")

    if patient_metrics is not None:
        print("\n" + "=" * 80)
        print("SIMILAR-PATIENT RETRIEVAL METRICS")
        print("=" * 80)
        for k, v in patient_metrics.items():
            print(f"{k}: {v:.4f}")

    print("\n" + "=" * 80)
    print("TOP RERANKED ARTICLES")
    print("=" * 80)
    for item in reranked_articles:
        print(
            f"[{item['rerank_rank']}] PMID={item['pmid']} | "
            f"dense={item['score']:.4f} | "
            f"bm25={item.get('bm25_score', 0.0):.4f} | "
            f"rerank={item['rerank_score']:.4f}"
        )
        print(f"Title: {item['article_title']}")
        print()

    print("\n" + "=" * 80)
    print("TOP RERANKED SIMILAR PATIENTS")
    print("=" * 80)
    for item in reranked_patients:
        print(
            f"[{item['rerank_rank']}] patient_uid={item['patient_uid']} | "
            f"dense={item['score']:.4f} | "
            f"bm25={item.get('bm25_score', 0.0):.4f} | "
            f"rerank={item['rerank_score']:.4f}"
        )
        print(f"Source title: {item['source_title']}")
        print()

    print("\n" + "=" * 80)
    print("REASONING PROMPT PREVIEW")
    print("=" * 80)
    print(truncate_text(reasoning_prompt, max_chars=4000))

    print("\n" + "=" * 80)
    print("LLM REASONING OUTPUT")
    print("=" * 80)
    print(formatted_reasoning_output)

    # -------------------------
    # TEXT REPORT ONLY
    # -------------------------
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("QUERY")
    report_lines.append("=" * 80)
    report_lines.append(truncate_text(query_text, max_chars=600))
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RETRIEVAL QUERY")
    report_lines.append("=" * 80)
    report_lines.append(retrieval_query)
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("EVIDENCE PACK SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Articles: {len(evidence_pack['articles'])}")
    report_lines.append(f"Similar patients: {len(evidence_pack['similar_patients'])}")
    if exclude_patient_uid:
        report_lines.append(f"Excluded patient_uid: {exclude_patient_uid}")
    if exclude_source_pmid:
        report_lines.append(f"Excluded source_pmid: {exclude_source_pmid}")
    report_lines.append("")

    if article_metrics is not None:
        report_lines.append("=" * 80)
        report_lines.append("ARTICLE RETRIEVAL METRICS")
        report_lines.append("=" * 80)
        for k, v in article_metrics.items():
            report_lines.append(f"{k}: {v:.4f}")
        report_lines.append("")

    if patient_metrics is not None:
        report_lines.append("=" * 80)
        report_lines.append("SIMILAR-PATIENT RETRIEVAL METRICS")
        report_lines.append("=" * 80)
        for k, v in patient_metrics.items():
            report_lines.append(f"{k}: {v:.4f}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TOP RERANKED ARTICLES")
    report_lines.append("=" * 80)
    for item in reranked_articles:
        report_lines.append(
            f"[{item['rerank_rank']}] PMID={item['pmid']} | "
            f"dense={item['score']:.4f} | "
            f"bm25={item.get('bm25_score', 0.0):.4f} | "
            f"rerank={item['rerank_score']:.4f}"
        )
        report_lines.append(f"Title: {item['article_title']}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TOP RERANKED SIMILAR PATIENTS")
    report_lines.append("=" * 80)
    for item in reranked_patients:
        report_lines.append(
            f"[{item['rerank_rank']}] patient_uid={item['patient_uid']} | "
            f"dense={item['score']:.4f} | "
            f"bm25={item.get('bm25_score', 0.0):.4f} | "
            f"rerank={item['rerank_score']:.4f}"
        )
        report_lines.append(f"Source title: {item['source_title']}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("REASONING PROMPT PREVIEW")
    report_lines.append("=" * 80)
    report_lines.append(truncate_text(reasoning_prompt, max_chars=4000))
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("LLM REASONING OUTPUT")
    report_lines.append("=" * 80)
    report_lines.append(formatted_reasoning_output)
    report_lines.append("")

    report_text = "\n".join(report_lines)

    report_path = Path(args.save_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved text report to: {report_path}")


if __name__ == "__main__":
    main()