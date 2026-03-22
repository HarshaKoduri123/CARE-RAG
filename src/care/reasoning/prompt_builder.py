from __future__ import annotations

from typing import Dict, List


class PromptBuilder:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.prompt_cfg = cfg["prompt"]

    def _format_articles(self, articles: List[Dict]) -> str:
        if not articles:
            return "No relevant articles were retrieved."

        lines = []
        for item in articles:
            line = (
                f"[Article {item['rank']}] "
                f"PMID: {item.get('pmid', '')}\n"
                f"Title: {item.get('title', '')}\n"
                f"Journal: {item.get('journal', '')} | Year: {item.get('pub_year', '')}\n"
            )

            if self.prompt_cfg.get("include_scores", True):
                line += (
                    f"Dense score: {item.get('dense_score', None)} | "
                    f"Rerank score: {item.get('rerank_score', None)}\n"
                )

            line += f"Abstract: {item.get('abstract', '')}\n"
            lines.append(line)

        return "\n".join(lines)

    def _format_similar_patients(self, patients: List[Dict]) -> str:
        if not patients:
            return "No similar patients were retrieved."

        lines = []
        for item in patients:
            line = (
                f"[Similar Patient {item['rank']}] "
                f"Patient UID: {item.get('patient_uid', '')}\n"
                f"Source PMID: {item.get('source_pmid', '')}\n"
                f"Source Title: {item.get('source_title', '')}\n"
            )

            if self.prompt_cfg.get("include_scores", True):
                line += (
                    f"Dense score: {item.get('dense_score', None)} | "
                    f"Rerank score: {item.get('rerank_score', None)}\n"
                )

            line += f"Patient Summary: {item.get('patient_text', '')}\n"
            lines.append(line)

        return "\n".join(lines)

    def build_reasoning_prompt(self, evidence_pack: Dict) -> str:
        query_text = evidence_pack["query_text"]
        articles = evidence_pack.get("articles", [])
        similar_patients = evidence_pack.get("similar_patients", [])

        sections = []
        sections.append(
            "You are an evidence-grounded clinical decision support assistant."
        )
        sections.append(
            "Your task is to analyze the patient case, compare it with similar historical cases, "
            "and use the retrieved biomedical literature to produce a careful, grounded clinical reasoning summary."
        )
        sections.append(
            "Do not fabricate evidence. Only use the provided patient case, similar cases, and retrieved literature."
        )
        sections.append(
            "Clearly separate observations from inferences. Where appropriate, cite supporting article PMIDs and similar patient IDs."
        )

        sections.append("\n=== QUERY PATIENT CASE ===\n")
        sections.append(query_text)

        if self.prompt_cfg.get("include_articles", True):
            sections.append("\n=== RETRIEVED ARTICLES ===\n")
            sections.append(self._format_articles(articles))

        if self.prompt_cfg.get("include_similar_patients", True):
            sections.append("\n=== RETRIEVED SIMILAR PATIENTS ===\n")
            sections.append(self._format_similar_patients(similar_patients))

        sections.append(
            "\n=== TASK ===\n"
            "Provide a structured response with the following sections:\n"
            "1. Key clinical findings in the query patient.\n"
            "2. Most relevant evidence from the retrieved articles.\n"
            "3. Comparison with the most relevant similar patients.\n"
            "4. Plausible clinical interpretation or differential considerations.\n"
            "5. Evidence-grounded recommendations or next considerations.\n"
            "6. Explicit evidence trace listing which PMIDs and similar patient IDs support the reasoning.\n"
        )

        return "\n".join(sections)