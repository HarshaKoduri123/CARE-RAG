from __future__ import annotations

import re
from typing import Callable, Dict

from care.utils.text import clean_text, truncate_text


class LLMClinicalQueryBuilder:
    def __init__(self, cfg: Dict, llm_generate_fn: Callable[[str], str]) -> None:
        self.cfg = cfg
        self.llm_generate_fn = llm_generate_fn

        qb_cfg = cfg.get("query_builder", {})
        self.max_query_chars = qb_cfg.get("max_query_chars", 400)
        self.fallback_to_raw = qb_cfg.get("fallback_to_raw", True)

    def build_prompt(self, patient_text: str) -> str:
        patient_text = clean_text(patient_text)

        return f"""
You are a clinical retrieval query generator.

Task:
Convert the following long patient case into a short retrieval query for biomedical literature and similar-patient retrieval.

Rules:
1. Output only a compact keyword-style retrieval query.
2. Do not write full sentences.
3. Do not include generic words such as admitted, appearance, history, started with, treated with.
4. Prioritize:
   - diagnoses
   - complications
   - organisms
   - drugs
   - devices
   - procedures
   - anatomy
5. Use only information explicitly present in the case.
6. Do not invent diagnoses or facts.
7. Keep the query under 40 words if possible.
8. Prefer comma-separated clinical concepts.

Patient case:
\"\"\"{patient_text}\"\"\"

Return only the retrieval query.
""".strip()

    def _postprocess(self, text: str) -> str:
        text = clean_text(text)
        text = re.sub(r"^(retrieval query|query)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = text.strip("`\"' ")
        return truncate_text(text, max_chars=self.max_query_chars)

    def build(self, patient_text: str) -> str:
        prompt = self.build_prompt(patient_text)
        raw_output = self.llm_generate_fn(prompt)
        query = self._postprocess(raw_output)

        if not query and self.fallback_to_raw:
            return truncate_text(clean_text(patient_text), max_chars=self.max_query_chars)

        return query