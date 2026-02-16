# AskHQ – Daily Progress Log

---

## Day 1 – 2026-02-16

**Phase:** Phase 0 – Environment + Infra

**Goal:** Runnable project scaffold with /health endpoint working.

**Planned:**
- [x] Create project structure (all dirs, __init__.py files)
- [x] Pin Python 3.11.9 via .python-version
- [x] Create requirements.txt + requirements-dev.txt
- [x] Create .env.example, Makefile
- [x] Implement app/config.py (pydantic-settings)
- [x] Implement app/main.py (FastAPI + middleware)
- [x] Implement GET /health endpoint
- [x] Write sample knowledge/policies/leave-policy.md
- [x] Implement RAG pipeline (chunker, embedder, vectorstore, retriever)
- [x] Implement Ollama LLM client + RAG service
- [x] Implement POST /ask + POST /reindex endpoints
- [x] Create reindex script (scripts/reindex.py)
- [x] Write tests (chunker, retriever, integration)
- [x] Write README with full setup instructions
- [x] Create PROGRESS.md + CHECKLIST.md
- [x] Git commit

**Completed:**
- [x] Full scaffold implemented

**Blockers:**
- None

**Decisions Made:**
- No env prefix (plain `OLLAMA_MODEL` instead of `ASKHQ_OLLAMA_MODEL`) — less friction
- Regex-based heading splitting instead of markdown-it-py AST — simpler, 40 lines vs fighting an API
- Max ~180 words per chunk (~230 tokens) — safely under MiniLM's 256-token limit
- No overlap across heading boundaries — headings are semantic boundaries
- Skip LLM entirely when no chunks pass threshold — eliminates hallucination vector

**Next (Day 2):**
- [ ] `make setup` → verify full env bootstrap
- [ ] `make reindex` → verify leave-policy.md chunks correctly
- [ ] `make dev` → test /health, /ask, /reindex with curl
- [ ] Tune MIN_SIMILARITY threshold with test queries
- [ ] Add 1-2 more knowledge docs

---

## Day N – YYYY-MM-DD

**Phase:** Phase X – [Name]

**Goal:** [One sentence]

**Planned:**
- [ ] Task 1
- [ ] Task 2

**Completed:**
- [x] Task 1 – [note]

**Not Completed:**
- [ ] Task 2 – [reason]

**Blockers:**
- [Description] / None

**Decisions Made:**
- Decided X over Y because [reason]

**Observations:**
- [Debugging notes, config that worked, errors encountered]

**Next (Day N+1):**
- [ ] Carry-over tasks
- [ ] New tasks
