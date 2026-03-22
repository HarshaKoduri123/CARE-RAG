from __future__ import annotations

import re
from typing import Dict, List

import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

from care.utils.text import clean_text, truncate_text


class ClinicalQueryBuilder:
    def __init__(self, cfg: Dict) -> None:
        qb_cfg = cfg["query_builder"]

        self.enabled = qb_cfg.get("enabled", True)
        self.model_name = qb_cfg.get("model", "en_core_sci_sm")
        self.use_abbreviation_detector = qb_cfg.get("use_abbreviation_detector", True)
        self.use_linker = qb_cfg.get("use_linker", False)
        self.linker_name = qb_cfg.get("linker_name", "umls")
        self.resolve_abbreviations = qb_cfg.get("resolve_abbreviations", True)
        self.max_entities = qb_cfg.get("max_entities", 20)
        self.max_query_chars = qb_cfg.get("max_query_chars", 400)
        self.min_entity_len = qb_cfg.get("min_entity_len", 2)
        self.deduplicate_case_insensitive = qb_cfg.get("deduplicate_case_insensitive", True)

        self.nlp = spacy.load(self.model_name)

        if self.use_abbreviation_detector and "abbreviation_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("abbreviation_detector")

        if self.use_linker and "scispacy_linker" not in self.nlp.pipe_names:
            self.nlp.add_pipe(
                "scispacy_linker",
                config={
                    "linker_name": self.linker_name,
                    "resolve_abbreviations": self.resolve_abbreviations,
                },
            )

        self.generic_stop_phrases = {
            "medical history",
            "history",
            "admitted",
            "appearance",
            "extended",
            "anterior",
            "subtotal",
            "removal",
            "surgical step",
            "step",
            "man",
            "woman",
            "boy",
            "girl",
            "caucasian",
            "patient",
            "day",
        }

    def _normalize_entity(self, text: str) -> str:
        text = clean_text(text)
        text = text.strip(" ,;:.()[]{}")
        return text

    def _is_abbreviation_like(self, text: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9\-]{2,10}", text))

    def _is_likely_microbe(self, text: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]?[a-z]+ [a-z0-9\-]+", text))

    def _is_useful_entity(self, text: str) -> bool:
        lower = text.lower()

        if len(text) < self.min_entity_len:
            return False

        if lower in self.generic_stop_phrases:
            return False

        # keep abbreviations like EVD, ELD, CSF
        if self._is_abbreviation_like(text):
            return True

        # keep multiword mentions
        if " " in text and len(text) >= 5:
            return True

        # keep likely microbes or medically useful suffixes
        useful_suffixes = (
            "itis", "oma", "emia", "pathy", "plasty", "ectomy", "otomy",
            "ostomy", "leak", "drain", "shunt", "edema", "oedema"
        )
        if lower.endswith(useful_suffixes):
            return True

        # keep drug-like / biomedical morphology
        useful_substrings = (
            "cef", "cillin", "mycin", "floxacin", "bactam", "colistin",
            "polymyxin", "amikacin", "pseudomonas", "acinetobacter",
            "staphyl", "enterococcus", "ventric", "lumbar", "pharyngeal",
            "sphenoid", "cliv", "chordoma", "hydrocephal", "meningit", "csf"
        )
        if any(s in lower for s in useful_substrings):
            return True

        # reject short generic single words
        if " " not in text and len(text) <= 4:
            return False

        return False

    def _score_entity(self, text: str) -> int:
        lower = text.lower()
        score = 0

        if self._is_abbreviation_like(text):
            score += 4
        if " " in text:
            score += 3
        if self._is_likely_microbe(text):
            score += 4

        priority_substrings = (
            "chordoma", "pseudomonas", "meningit", "hydrocephal", "csf",
            "drain", "ventric", "lumbar", "sphenoid", "cliv", "resection",
            "approach", "leak", "fistula", "cef", "floxacin", "amikacin",
            "polymyxin"
        )
        for s in priority_substrings:
            if s in lower:
                score += 3

        if lower.endswith(("itis", "oma", "ectomy", "otomy", "leak")):
            score += 2

        return score

    def _collect_mentions(self, text: str) -> List[str]:
        doc = self.nlp(text)
        mentions: List[str] = []

        for ent in doc.ents:
            ent_text = self._normalize_entity(ent.text)
            if self._is_useful_entity(ent_text):
                mentions.append(ent_text)

        if self.use_abbreviation_detector and hasattr(doc._, "abbreviations"):
            for abrv in doc._.abbreviations:
                short_form = self._normalize_entity(str(abrv))
                if self._is_useful_entity(short_form):
                    mentions.append(short_form)

                long_form = getattr(abrv._, "long_form", None)
                if long_form is not None:
                    long_form_text = self._normalize_entity(str(long_form))
                    if self._is_useful_entity(long_form_text):
                        mentions.append(long_form_text)

        return mentions

    def _deduplicate(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()

        for item in items:
            key = item.lower() if self.deduplicate_case_insensitive else item
            if key not in seen:
                seen.add(key)
                out.append(item)

        return out

    def build(self, patient_text: str) -> str:
        text = clean_text(patient_text)

        if not self.enabled:
            return truncate_text(text, max_chars=self.max_query_chars)

        mentions = self._collect_mentions(text)
        mentions = self._deduplicate(mentions)
        mentions.sort(key=self._score_entity, reverse=True)
        mentions = mentions[: self.max_entities]

        if not mentions:
            return truncate_text(text, max_chars=self.max_query_chars)

        query = ", ".join(mentions)
        return truncate_text(query, max_chars=self.max_query_chars)