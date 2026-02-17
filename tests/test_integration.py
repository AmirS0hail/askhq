"""Integration tests for the FastAPI endpoints.

Uses TestClient with mocked RAG dependencies (no Ollama/Chroma needed).
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── GET /health ──────────────────────────────────────────────


def test_health_endpoint_returns_200():
    """Health endpoint should always return 200 (even if degraded)."""
    mock_provider = AsyncMock()
    mock_provider.check_health.return_value = True

    with (
        patch("app.api.routes.get_llm_provider", return_value=mock_provider),
        patch("app.api.routes.get_chroma_store") as mock_store,
    ):
        mock_store.return_value.count.return_value = 42
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["llm_connected"] is True
    assert data["chroma_docs"] == 42


def test_health_degraded_when_ollama_down():
    mock_provider = AsyncMock()
    mock_provider.check_health.return_value = False

    with (
        patch("app.api.routes.get_llm_provider", return_value=mock_provider),
        patch("app.api.routes.get_chroma_store") as mock_store,
    ):
        mock_store.return_value.count.return_value = 10
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["llm_connected"] is False


# ── POST /ask ────────────────────────────────────────────────


def test_ask_returns_answer():
    mock_result = {
        "answer": "You get 3 casual leave days per year.",
        "sources": [
            {
                "doc_name": "leave-policy.md",
                "doc_path": "policies/leave-policy.md",
                "section": "Leave Policy > Types of Leave > Casual Leave",
                "chunk_id": "policies/leave-policy.md#chunk1",
                "similarity_score": 0.89,
            }
        ],
        "query": "How many casual leave days?",
        "retrieved_chunks": 1,
        "processing_time_ms": 123.4,
    }

    with patch("app.api.routes.rag_service.answer_question", new_callable=AsyncMock, return_value=mock_result):
        response = client.post("/ask", json={"question": "How many casual leave days?"})

    assert response.status_code == 200
    data = response.json()
    assert "casual leave" in data["answer"].lower()
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_name"] == "leave-policy.md"


def test_ask_fallback_when_no_context():
    mock_result = {
        "answer": "I couldn't find this in the current policy documents. Please check with Alfabolt HR.",
        "sources": [],
        "query": "What is the dress code?",
        "retrieved_chunks": 0,
        "processing_time_ms": 15.2,
    }

    with patch("app.api.routes.rag_service.answer_question", new_callable=AsyncMock, return_value=mock_result):
        response = client.post("/ask", json={"question": "What is the dress code?"})

    assert response.status_code == 200
    data = response.json()
    assert "couldn't find" in data["answer"].lower() or "check with" in data["answer"].lower()
    assert data["sources"] == []


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_long_question():
    response = client.post("/ask", json={"question": "x" * 501})
    assert response.status_code == 422


# ── POST /reindex ────────────────────────────────────────────


def test_reindex_requires_token():
    response = client.post("/reindex")
    assert response.status_code == 422  # missing header


def test_reindex_rejects_bad_token():
    response = client.post("/reindex", headers={"X-Reindex-Token": "wrong-token"})
    assert response.status_code == 403


def test_reindex_success():
    with (
        patch("app.api.routes.chunk_all_documents") as mock_chunk,
        patch("app.api.routes.get_embedding_service") as mock_embed,
        patch("app.api.routes.get_chroma_store") as mock_store,
    ):
        # Mock chunking
        from app.rag.chunker import Chunk

        fake_chunks = [
            Chunk(id="test.md#chunk0", text="Test content", metadata={"doc_path": "test.md"}),
        ]
        mock_chunk.return_value = fake_chunks

        # Mock embedding
        import numpy as np

        mock_embed.return_value.embed_documents.return_value = np.array([[0.1] * 384])

        # Mock store
        mock_store.return_value.add_chunks.return_value = 1

        response = client.post(
            "/reindex",
            headers={"X-Reindex-Token": "changeme-local-dev-token"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] == 1
