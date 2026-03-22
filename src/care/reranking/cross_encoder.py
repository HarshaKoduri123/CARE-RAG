from __future__ import annotations

from typing import List, Sequence, Tuple

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[float]:
        if not pairs:
            return []
        scores = self.model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [float(x) for x in scores]