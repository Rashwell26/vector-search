# Vector Search

A small semantic search engine built in Python. It turns documents into
embedding vectors, stores them in an index, and answers queries with
cosine-similarity top-k search — accessible from a web UI, a command line, and a REST API.

## How it works

```
documents (.txt)  ──►  embedder (sentence-transformers)  ──►  vectors
                                                                      │
query text  ──►  embedder  ──►  query vector  ──►  cosine similarity ──► top-k results
```

1. **Ingest** — every `.txt` file in `data/` becomes a document.
2. **Embed** — `all-MiniLM-L6-v2` converts each document into a 384-dimensional unit vector, so similar texts end up close together.
3. **Store** — vectors and metadata are persisted to `index/` (`vectors.npz` + `documents.json`).
4. **Search** — the query is embedded the same way, compared with a dot product (equal to cosine similarity for unit vectors), and the top-k matches are returned.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # on Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python scripts/make_sample_data.py   # generate the 55 sample documents
python index.py                       # build the index (downloads the ~90MB model on first run)
```

## Usage

**Command line**

```bash
python cli.py "how do database indexes work"
python cli.py "what causes a deadlock" --k 3
```

**Web UI + API**

```bash
uvicorn api:app --reload
```

Then open <http://127.0.0.1:8000> for the search interface, or
<http://127.0.0.1:8000/docs> for interactive API docs.

```bash
curl "http://127.0.0.1:8000/search?q=graph+shortest+path&k=3"
curl "http://127.0.0.1:8000/health"
```

**Tests** (no model download needed — they use a fake embedder)

```bash
pytest
```

## Project layout

```
vector-search/
├── vector_search/          # core library
│   ├── embedder.py         # text -> vectors (sentence-transformers)
│   ├── store.py            # document + vector storage with save/load
│   └── search.py           # cosine similarity top-k search
├── data/
│   ├── docs/               # 50 general computer science documents
│   └── research/           # 5 niche research topics
├── index.py                # builds the index from data/
├── cli.py                  # command-line interface
├── api.py                  # FastAPI REST API + serves the web UI
├── static/                 # web frontend (HTML/CSS/JS, no build step)
├── scripts/                # sample data generator
└── tests/                  # pytest suite
```

## Possible extensions

- Swap `all-MiniLM-L6-v2` for a larger model or try TF-IDF from scratch
- Add hybrid search that combines keyword (BM25) and vector scores
- Index your own documents by dropping `.txt` files into `data/`
- Containerize with Docker for one-command deployment
