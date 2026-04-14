# Graph RAG vs Vector RAG -- Comparison System

A side-by-side comparison of two Retrieval-Augmented Generation approaches:

- **Graph RAG** -- LlamaIndex PropertyGraphIndex with Neo4j for entity/relation extraction and graph-based retrieval.
- **Vector RAG** -- OpenRouter embeddings with ChromaDB for cosine-similarity chunk retrieval.

Both pipelines share the same LLM (via OpenRouter) for answer generation. A Gradio frontend displays results from both systems simultaneously.

---

## Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- OpenRouter API key

## Installation

### 1. Install uv (package manager)

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

---

## Configuration

```bash
cp .env_example .env
```

Edit `.env` and set your OpenRouter API key:

```
OPENAI_API_KEY="sk-or-..."
```

Other settings (with defaults):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenRouter API key |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `CHROMA_PERSIST_DIR` | (empty = in-memory) | ChromaDB persistence path |

---

## Running

### 1. Start Neo4j

```bash
docker compose up -d
```

Neo4j browser: http://localhost:7474 (login: `neo4j` / `password`)

### 2. Start the backend

```bash
uv run backend
```

API available at http://localhost:8002. Swagger docs at http://localhost:8002/docs.

### 3. Start the frontend

```bash
uv run frontend
```

UI available at http://localhost:7860.

---

## Usage

1. Open http://localhost:7860.
2. If no dataset exists, open **Settings** and use **Wiki Dataset Collection** to gather articles on a topic.
3. Type a question. Both Graph RAG and Vector RAG answer simultaneously in side-by-side panels.
4. Sources with relevance scores are displayed below each answer.
5. Adjust **Top-K**, **Temperature**, and **LLM Model** in Settings.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query/dual` | Query both RAG systems in parallel |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System statistics (graph + vector) |
| `POST` | `/collect-wiki` | Start Wikipedia dataset collection |
| `GET` | `/collect-wiki/status` | Poll collection progress |

---

## Project Structure

```
app/
  backend.py           FastAPI server, LLM generation, dual query orchestration
  frontend.py          Gradio UI with side-by-side comparison
  graph_rag.py         Graph RAG: LlamaIndex PropertyGraphIndex + Neo4j
  vector_store.py      Vector RAG: ChromaDB + OpenRouter embeddings
  document_processor.py  Document loading and chunking
  dataset/
    data_from_wiki.txt   Collected Wikipedia articles
experiments/
  collect_wiki.py      Wikipedia collection script with LLM topic expansion
  quality_assessment.py  Automated quality comparison
  rag_comparison.py    RAG pipeline comparison utilities
docker-compose.yml     Neo4j container configuration
pyproject.toml         Project metadata and dependencies
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of how the system works.

---

## Resetting Neo4j

To clear all graph data and start fresh:

```bash
docker compose down -v
docker compose up -d
```

---

## Troubleshooting

- **Cannot connect to backend** -- ensure `uv run backend` is running in a separate terminal.
- **OPENAI_API_KEY not found** -- check your `.env` file.
- **Neo4j connection failed** -- verify Docker is running: `docker compose up -d`, then check http://localhost:7474.
- **Embedding errors** -- confirm your OpenRouter key has access to embedding models.
- **Graph build takes too long** -- the first build extracts entities via LLM. Subsequent starts reuse cached data if the dataset has not changed.
