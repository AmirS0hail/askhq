.PHONY: setup-python venv install setup dev run reindex test lint format clean health

# ── Environment setup ───────────────────────────────────────

setup-python:
	@echo "Checking pyenv..."
	@command -v pyenv >/dev/null 2>&1 || { echo "❌ pyenv not found. Install: brew install pyenv"; exit 1; }
	pyenv install -s 3.11.9
	pyenv local 3.11.9
	@echo "✅ Python 3.11.9 ready"

venv:
	python -m venv .venv
	@echo "✅ Virtual environment created at .venv/"
	@echo "   Activate: source .venv/bin/activate"

install:
	@test -d .venv || { echo "❌ No .venv found. Run 'make venv' first."; exit 1; }
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

setup: setup-python venv install
	@test -f .env || cp .env.example .env
	@echo ""
	@echo "✅ Setup complete. Next steps:"
	@echo "   1. ollama pull llama3.1:8b"
	@echo "   2. Add docs to knowledge/"
	@echo "   3. make reindex"
	@echo "   4. make dev"

# ── Development ─────────────────────────────────────────────

dev:
	.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

reindex:
	.venv/bin/python -m scripts.reindex

# ── Quality ─────────────────────────────────────────────────

test:
	.venv/bin/pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	.venv/bin/ruff check app/ scripts/ tests/
	.venv/bin/mypy app/ --ignore-missing-imports

format:
	.venv/bin/ruff format app/ scripts/ tests/

# ── Utilities ───────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

health:
	@echo "── API ──"
	@curl -sf http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "❌ API not responding"
	@echo ""
	@echo "── Ollama ──"
	@curl -sf http://localhost:11434/api/tags | python -m json.tool 2>/dev/null || echo "❌ Ollama not responding"
