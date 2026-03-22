from __future__ import annotations

from typing import Dict

from care.llm.ollama_client import OllamaClient


class LLMGenerator:
    def __init__(self, cfg: Dict) -> None:
        llm_cfg = cfg["reasoning_llm"]
        provider = llm_cfg.get("provider", "ollama")

        if provider != "ollama":
            raise ValueError(f"Unsupported reasoning_llm provider: {provider}")

        self.client = OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model", "llama3.1:8b"),
            timeout_seconds=llm_cfg.get("timeout_seconds", 180),
        )
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.max_tokens = llm_cfg.get("max_tokens", 1200)

    def generate(self, prompt: str) -> str:
        return self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )