"""Vector Search — a small semantic search engine over text documents."""

from vector_search.embedder import Embedder
from vector_search.search import SearchResult, search
from vector_search.store import Document, VectorStore

__all__ = ["Document", "Embedder", "SearchResult", "VectorStore", "search"]
