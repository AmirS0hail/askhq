# AskHQ – Alfabolt HR Policy Assistant

AI-powered assistant that answers employee questions about company policies using official documents only. Built for a 2-week hackathon with zero-hallucination design.

**How it works:** Markdown policy docs → chunked & embedded → stored in Chroma → user asks question → retrieve relevant chunks → LLM answers strictly from context → if no context found, says "check with HR".

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11 (pyenv) | Stable, typed, fast iteration |
| API | FastAPI | Async, auto-docs, Pydantic native |
| Vector DB | Chroma (local) | Zero config, persistent, good enough |
| Embeddings | all-MiniLM-L6-v2 | Free, local, 256-token, 384-dim |
| LLM | Ollama (local) | Free, no API keys, runs llama3.1/mistral |
| Knowledge | Markdown files | Simple, versionable, no API needed |

## Prerequisites

- **macOS or Linux**
- **pyenv** – Python version management
- **Ollama** – Local LLM inference

### Install pyenv (skip if installed)

```bash
# macOS
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Linux
curl https://pyenv.run | bash
# Follow printed instructions to add to shell rc
```

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

## Quick Start

```bash
# 1. Clone
git clone <repo-url> askhq && cd askhq

# 2. Setup Python + venv + dependencies (one command)
make setup

# 3. Start Ollama and pull model
ollama serve &          # run in background
ollama pull llama3.1:8b # ~4.7 GB download

# 4. Configure
cp .env.example .env    # edit if needed, defaults work locally

# 5. Index knowledge base
make reindex

# 6. Start server
make dev
# → http://localhost:8000
# → http://localhost:8000/docs (Swagger UI)

# 7. Test
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many casual leave days do I get?"}'
```

## Development Environment

**Stack:** pyenv + stdlib venv + pip + requirements.txt. No Poetry, no Conda, no Docker (yet). Optimized for hackathon speed and zero onboarding friction.

### What's committed vs gitignored

| File | Committed | Purpose |
|------|-----------|---------|
| `.python-version` | ✅ | pyenv reads this → team uses same Python |
| `requirements.txt` | ✅ | Pinned production dependencies |
| `requirements-dev.txt` | ✅ | Test/lint tooling |
| `.env.example` | ✅ | Template for local config |
| `.venv/` | ❌ | Per-machine virtual environment |
| `.env` | ❌ | Local secrets/config |
| `data/chroma/` | ❌ | Per-machine vector DB |

### Manual setup (if not using make)

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

## API Reference

### GET /health

Check system health.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "ollama_connected": true,
  "chroma_docs": 12,
  "version": "0.1.0"
}
```

### POST /ask

Ask a question. The Slack bot calls this endpoint.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maternity leave policy?"}'
```

```json
{
  "answer": "Maternity leave at Alfabolt is 3 months (90 calendar days) of paid leave...",
  "sources": [
    {
      "doc_name": "leave-policy.md",
      "doc_path": "policies/leave-policy.md",
      "section": "Leave Policy > Special Leave > Maternity Leave",
      "chunk_id": "policies/leave-policy.md#chunk5",
      "similarity_score": 0.91
    }
  ],
  "query": "What is the maternity leave policy?",
  "retrieved_chunks": 3,
  "processing_time_ms": 2450.0
}
```

Optional parameters: `top_k` (int, 1–20), `min_score` (float, 0.0–1.0).

### POST /reindex

Rebuild vector database. Protected by token.

```bash
curl -X POST http://localhost:8000/reindex \
  -H "X-Reindex-Token: changeme-local-dev-token"
```

```json
{
  "status": "success",
  "documents_processed": 1,
  "chunks_created": 12,
  "duration_s": 3.45
}
```

## Adding Knowledge Documents

1. Create a markdown file in `knowledge/` (subdirectories OK):

```markdown
---
title: Benefits Policy
category: hr-policies
version: "1.0"
last_updated: "2026-02-01"
---

# Benefits Policy

## Health Insurance

All employees are covered under group health insurance...
```

2. Run `make reindex` to update the vector database.
3. Test with `curl` or Swagger UI.

**Tips:**
- Use H2+ headings to create logical sections (each becomes a chunk)
- Keep sections under ~180 words for best retrieval quality
- YAML frontmatter is optional but helps with source attribution
- The chunker splits on headings and further splits long sections by paragraphs

## Slack Integration

The Slack bot is a separate service. It sends questions to this API's `/ask` endpoint.

### Local development with ngrok

```bash
# Terminal 1: Start API
make dev

# Terminal 2: Expose via ngrok
ngrok http 8000
# Copy the HTTPS URL, e.g. https://abc123.ngrok-free.app

# Configure Slack bot to POST to:
# https://abc123.ngrok-free.app/ask
```

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Full first-time setup (Python + venv + deps) |
| `make dev` | Start server with hot reload (port 8000) |
| `make run` | Start production server (2 workers) |
| `make reindex` | Rebuild vector database from /knowledge |
| `make test` | Run pytest with coverage |
| `make lint` | Ruff + mypy checks |
| `make format` | Auto-format code with ruff |
| `make clean` | Remove Python cache files |
| `make health` | Check API + Ollama status |

## Project Structure

```
askhq/
├── .python-version          # pyenv pin: 3.11.9
├── app/
│   ├── main.py              # FastAPI app + middleware
│   ├── config.py            # Settings from .env
│   ├── api/routes.py        # HTTP endpoints
│   ├── rag/
│   │   ├── chunker.py       # Markdown → chunks
│   │   ├── embedder.py      # SentenceTransformer singleton
│   │   ├── vectorstore.py   # Chroma wrapper
│   │   ├── retriever.py     # Search + threshold filter
│   │   ├── llm_client.py    # Ollama HTTP client
│   │   └── rag_service.py   # Retrieve → prompt → answer
│   └── models/schemas.py    # Pydantic models
├── knowledge/               # Markdown docs (the knowledge base)
│   └── policies/
├── data/                    # Runtime data (gitignored)
│   ├── chroma/              # Vector DB
│   └── chunks.jsonl         # Debug: chunker output
├── scripts/
│   ├── reindex.py           # CLI: rebuild vector DB
│   └── test_ollama.py       # Check Ollama connectivity
└── tests/                   # pytest suite
```

## Troubleshooting

**"Ollama not running"** – Start it: `ollama serve`

**"Model not found"** – Pull it: `ollama pull llama3.1:8b`

**"No results found" for valid questions** – Lower threshold in `.env`: `MIN_SIMILARITY=0.3` and reindex: `make reindex`

**"Module not found" errors** – Activate venv: `source .venv/bin/activate`

**Chroma corrupted** – Reset: `rm -rf data/chroma && make reindex`

**Empty answers** – Check: `curl localhost:8000/health` → `chroma_docs` should be > 0

## License

MIT