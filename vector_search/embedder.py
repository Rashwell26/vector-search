"""Turns text into embedding vectors using a sentence-transformers model."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformers model.

    Vectors are normalized (unit length) so that cosine similarity
    between two texts equals a simple dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Imported lazily so tests can use a fake embedder without
        # downloading the model.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length vector per input text, shape (n, 384)."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
