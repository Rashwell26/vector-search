"""Cosine-similarity top-k search over a VectorStore."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_search.store import Document, VectorStore


@dataclass(frozen=True)
class SearchResult:
    """A single match, with its similarity score."""

    document: Document
    score: float


def search(store: VectorStore, query_vector: np.ndarray, k: int = 5) -> list[SearchResult]:
    """Return the k documents most similar to query_vector, best first.

    All vectors in the store are normalized, so cosine similarity is
    just the dot product of the query with every stored vector.
    """
    if store.vectors.shape[0] == 0:
        return []

    scores = store.vectors @ query_vector
    k = min(k, len(store))

    # argsort ascending, then flip so the best score comes first
    best_indices = np.argsort(scores)[::-1][:k]

    return [
        SearchResult(document=store.documents[i], score=float(scores[i]))
        for i in best_indices
    ]
