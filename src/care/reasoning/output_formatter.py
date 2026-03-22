from __future__ import annotations

from typing import Dict


class ReasoningOutputFormatter:
    def format_for_console(self, parsed_output: Dict) -> str:
        return parsed_output.get("raw_text", "").strip()