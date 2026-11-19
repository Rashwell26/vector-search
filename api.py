"""A small REST API for the search engine.

Run it:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for interactive docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vector_search.embedder import Embedder
from vector_search.search import search
from vector_search.store import VectorStore

INDEX_DIR = Path("index")
STATIC_DIR = Path(__file__).parent / "static"
MAX_K = 50


def create_app(store: VectorStore | None = None, embedder: Embedder | None = None) -> FastAPI:
    """Build the app. Tests pass in fakes; production loads from disk."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store if store is not None else VectorStore.load(INDEX_DIR)
        app.state.embedder = embedder if embedder is not None else Embedder()
        yield

    app = FastAPI(title="Vector Search API", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Serve the web frontend."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Simple health check: how many documents are indexed."""
        return {"status": "ok", "documents": len(app.state.store)}

    @app.get("/search")
    def run_search(q: str, k: int = 5) -> dict[str, Any]:
        """Return the k documents most similar to q."""
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")
        k = max(1, min(k, MAX_K))

        query_vector = app.state.embedder.embed([q])[0]
        results = search(app.state.store, query_vector, k=k)

        return {
            "query": q,
            "results": [
                {
                    "id": r.document.id,
                    "source": r.document.source,
                    "text": r.document.text,
                    "score": round(r.score, 4),
                }
                for r in results
            ],
        }

    return app


app = create_app()
