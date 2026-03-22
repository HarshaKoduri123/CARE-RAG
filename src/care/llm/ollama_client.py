from __future__ import annotations

from typing import Dict

import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> str:
        url = f"{self.base_url}/api/generate"

        payload: Dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()

        data = resp.json()
        return data.get("response", "").strip()