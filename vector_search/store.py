"""Stores documents together with their vectors, and saves/loads them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VECTORS_FILE = "vectors.npz"
DOCUMENTS_FILE = "documents.json"


@dataclass(frozen=True)
class Document:
    """One indexed text snippet."""

    id: int
    text: str
    source: str


class VectorStore:
    """Holds the documents and their embedding vectors in memory."""

    def __init__(self, documents: list[Document], vectors: np.ndarray) -> None:
        if len(documents) != vectors.shape[0]:
            raise ValueError(
                f"{len(documents)} documents but {vectors.shape[0]} vectors"
            )
        self.documents = documents
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.documents)

    def save(self, path: str | Path) -> None:
        """Persist the index to a folder (JSON metadata + NPZ vectors)."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        metadata = [
            {"id": d.id, "text": d.text, "source": d.source}
            for d in self.documents
        ]
        (target / DOCUMENTS_FILE).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        np.savez_compressed(target / VECTORS_FILE, vectors=self.vectors)

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Load an index previously written by save()."""
        source = Path(path)
        metadata = json.loads(
            (source / DOCUMENTS_FILE).read_text(encoding="utf-8")
        )
        documents = [Document(**item) for item in metadata]
        with np.load(source / VECTORS_FILE) as data:
            vectors = data["vectors"]
        return cls(documents, vectors)
