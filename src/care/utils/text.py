from __future__ import annotations

import re
from typing import Optional


_MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€“": "-",
    "â€”": "-",
    "Â°": "°",
    "Âµ": "µ",
    "Î¼": "μ",
    "Ã—": "×",
}


def fix_common_mojibake(text: str) -> str:
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = fix_common_mojibake(str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_chars: int = 2000) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."