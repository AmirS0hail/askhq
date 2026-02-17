# ── AskHQ: FastAPI + RAG + HuggingFace ──────────────────────
# Optimized for Render free tier (512 MB RAM)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system dependencies ─────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ─────────────────────────────
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model at build time ──────────────
# Avoids ~90 MB download on every cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ───────────────────────────────────
COPY app/ app/
COPY scripts/ scripts/
COPY knowledge/ knowledge/

# ── Make start script executable ─────────────────────────────
COPY scripts/start.sh scripts/start.sh
RUN chmod +x scripts/start.sh

# ── Expose port (Render sets $PORT dynamically) ─────────────
EXPOSE 8000

# ── Boot: reindex + start server ─────────────────────────────
CMD ["scripts/start.sh"]
