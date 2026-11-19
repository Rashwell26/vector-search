"""Tests for the search engine.

These use a fake embedder so they run in milliseconds without
downloading the sentence-transformers model.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app
from vector_search.search import search
from vector_search.store import Document, VectorStore

# Tiny vocabulary so the fake embedder produces deterministic vectors
VOCAB = ["database", "index", "python", "network", "sort", "vector", "search", "graph"]


class FakeEmbedder:
    """Turns text into a bag-of-words vector over VOCAB."""

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(len(VOCAB), dtype=np.float32)
            for token in text.lower().split():
                if token in VOCAB:
                    vector[VOCAB.index(token)] += 1
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            rows.append(vector)
        return np.vstack(rows)


@pytest.fixture
def store() -> VectorStore:
    embedder = FakeEmbedder()
    texts = [
        "python search engine over a vector index",
        "database index speeds up sql queries",
        "graph network routing algorithms",
        "sort an array with merge sort",
    ]
    documents = [Document(id=i, text=t, source=f"doc{i}.txt") for i, t in enumerate(texts)]
    return VectorStore(documents, embedder.embed(texts))


def test_save_and_load_roundtrip(tmp_path, store: VectorStore) -> None:
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    assert [d.text for d in loaded.documents] == [d.text for d in store.documents]
    np.testing.assert_array_almost_equal(loaded.vectors, store.vectors)


def test_mismatched_documents_and_vectors_rejected() -> None:
    with pytest.raises(ValueError):
        VectorStore([Document(id=0, text="a", source="a.txt")], np.zeros((2, 8)))


def test_search_returns_results_sorted_by_score(store: VectorStore) -> None:
    query = FakeEmbedder().embed(["database index"])[0]
    results = search(store, query, k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_ranks_matching_document_first(store: VectorStore) -> None:
    query = FakeEmbedder().embed(["python vector search"])[0]
    results = search(store, query, k=1)
    assert results[0].document.source == "doc0.txt"


def test_search_k_larger_than_store_returns_everything(store: VectorStore) -> None:
    query = FakeEmbedder().embed(["anything"])[0]
    assert len(search(store, query, k=100)) == len(store)


def test_search_on_empty_store_returns_nothing() -> None:
    empty = VectorStore([], np.zeros((0, 8), dtype=np.float32))
    query = np.zeros(8, dtype=np.float32)
    assert search(empty, query) == []


def test_api_search_endpoint(store: VectorStore) -> None:
    client = TestClient(create_app(store=store, embedder=FakeEmbedder()))
    with client:
        response = client.get("/search", params={"q": "python vector", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "python vector"
    assert len(body["results"]) == 2
    assert body["results"][0]["source"] == "doc0.txt"


def test_api_rejects_empty_query(store: VectorStore) -> None:
    client = TestClient(create_app(store=store, embedder=FakeEmbedder()))
    with client:
        response = client.get("/search", params={"q": "   "})
    assert response.status_code == 400


def test_api_health_endpoint(store: VectorStore) -> None:
    client = TestClient(create_app(store=store, embedder=FakeEmbedder()))
    with client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "documents": 4}


def test_home_page_serves_frontend(store: VectorStore) -> None:
    client = TestClient(create_app(store=store, embedder=FakeEmbedder()))
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Vector Search" in response.text


def test_static_assets_are_served(store: VectorStore) -> None:
    client = TestClient(create_app(store=store, embedder=FakeEmbedder()))
    with client:
        script = client.get("/static/app.js")
        stylesheet = client.get("/static/style.css")
    assert script.status_code == 200
    assert stylesheet.status_code == 200
