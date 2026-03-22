from __future__ import annotations

from typing import Dict


class ReasoningOutputParser:
    def parse(self, text: str) -> Dict:
        return {
            "raw_text": text.strip(),
        }