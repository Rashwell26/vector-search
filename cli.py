"""Search the index from the command line.

Usage:
    python cli.py "how do hash tables handle collisions"
    python cli.py "what is a deadlock" --k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vector_search.embedder import Embedder
from vector_search.search import search
from vector_search.store import VectorStore

INDEX_DIR = Path("index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over the document index")
    parser.add_argument("query", help="the text to search for")
    parser.add_argument("--k", type=int, default=5, help="number of results (default: 5)")
    parser.add_argument(
        "--index", type=Path, default=INDEX_DIR, help="index folder (default: index/)"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not (args.index / "documents.json").exists():
        raise SystemExit("No index found — run `python index.py` first")

    store = VectorStore.load(args.index)
    embedder = Embedder()
    query_vector = embedder.embed([args.query])[0]

    results = search(store, query_vector, k=args.k)

    print(f'\nQuery: "{args.query}"\n')
    for rank, result in enumerate(results, start=1):
        snippet = result.document.text[:120].replace("\n", " ")
        print(f"{rank}. [{result.score:.3f}] {result.document.source}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
