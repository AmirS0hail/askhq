# AskHQ – MVP Readiness Checklist

Last updated: 2026-02-16

## Environment

- [ ] `.python-version` committed (3.11.9)
- [ ] `python --version` → 3.11.x after `pyenv local`
- [ ] `.venv` created and activates cleanly
- [ ] `pip install -r requirements.txt` succeeds with no errors
- [ ] `.env.example` copied to `.env`
- [ ] `make dev` starts server on port 8000

## Ollama

- [ ] Ollama installed and `ollama serve` running
- [ ] Model pulled: `ollama list` shows configured model
- [ ] `curl http://localhost:11434/api/tags` returns 200

## Knowledge Base

- [ ] At least 1 markdown doc in `knowledge/` with YAML frontmatter
- [ ] `make reindex` completes without errors
- [ ] Reindex output shows chunk count > 0
- [ ] `data/chroma/` directory exists and is non-empty

## API Endpoints

- [ ] `GET /health` → 200, returns `{"status": "...", "ollama_connected": ..., "chroma_docs": N}`
- [ ] `GET /health` with Ollama stopped → 200, `"ollama_connected": false`
- [ ] `POST /ask` with valid question → 200, returns answer + sources
- [ ] `POST /ask` with out-of-scope question → 200, returns "contact HR" fallback
- [ ] `POST /ask` with empty question → 422 validation error
- [ ] `POST /ask` with >500 char question → 422 validation error
- [ ] `POST /ask` with Ollama stopped → 200, returns raw chunk excerpts (graceful degradation)
- [ ] `POST /reindex` without token → 422
- [ ] `POST /reindex` with bad token → 403
- [ ] `POST /reindex` with valid token → 200, returns stats

## RAG Quality

- [ ] In-scope query returns correct chunks (check scores in logs)
- [ ] Out-of-scope query returns no chunks above threshold
- [ ] Answer cites source document name and section
- [ ] Answer does NOT contain info absent from provided context
- [ ] Similarity scores are in [0, 1] range (distance→similarity conversion correct)
- [ ] Threshold filtering works: lowering `MIN_SIMILARITY` returns more results

## Hallucination Prevention

- [ ] Ask "What is the dress code?" (not in docs) → fallback, no fabrication
- [ ] Ask "How many vacation days at Google?" → fallback, no Alfabolt-specific fabrication
- [ ] Ask a partially matching question → answer covers only what's in docs
- [ ] LLM prompt includes "Answer ONLY using the provided context"
- [ ] When no chunks pass threshold, LLM is NOT called (verify via logs)

## Slack Integration

- [ ] ngrok tunnel active: `ngrok http 8000`
- [ ] Slack bot configured with ngrok HTTPS URL
- [ ] Slack message → bot → `/ask` → response posted in channel
- [ ] Response includes source references
- [ ] Fallback message displays correctly in Slack

## Demo Readiness

- [ ] Demo script: 5 questions (2 in-scope, 1 out-of-scope, 1 edge, 1 follow-up)
- [ ] Demo rehearsed end-to-end at least once
- [ ] Backup: screen recording if live demo fails
- [ ] README enables new person to run in <10 min

## Code Quality

- [ ] `make test` → all tests pass
- [ ] `make lint` → no errors
- [ ] No hardcoded secrets in committed code
- [ ] `.env` in `.gitignore`
- [ ] Public functions have docstrings
