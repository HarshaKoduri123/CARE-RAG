from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class BiEncoder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)