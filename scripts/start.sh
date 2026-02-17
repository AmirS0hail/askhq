#!/bin/sh
# Boot script for Render: reindex Chroma from baked-in knowledge/, then start server.
# Render sets $PORT dynamically — default to 8000 for local Docker testing.

set -e

PORT="${PORT:-8000}"

echo "=== AskHQ Boot ==="
echo "Reindexing knowledge base..."
python -m scripts.reindex --clean

echo "Starting server on port $PORT..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
