"""Build the vector index from the documents in data/.

Usage:
    python index.py
"""

from __future__ import annotations

from pathlib import Path

from vector_search.embedder import Embedder
from vector_search.store import Document, VectorStore

DATA_DIRS = [Path("data/docs"), Path("data/research")]
INDEX_DIR = Path("index")


def load_documents() -> list[Document]:
    """Read every .txt file in data/ into a list of Documents."""
    documents: list[Document] = []
    for folder in DATA_DIRS:
        for text_file in sorted(folder.glob("*.txt")):
            text = text_file.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    Document(id=len(documents), text=text, source=text_file.name)
                )
    return documents


def main() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit("No documents found in data/ — run scripts/make_sample_data.py")

    print(f"Embedding {len(documents)} documents...")
    embedder = Embedder()
    vectors = embedder.embed([d.text for d in documents])

    store = VectorStore(documents, vectors)
    store.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/ ({len(store)} documents, {vectors.shape[1]} dims)")


if __name__ == "__main__":
    main()
