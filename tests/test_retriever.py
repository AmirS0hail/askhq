"""Tests for the retriever (threshold filtering logic).

Uses mocks to avoid loading embedding models / Chroma in tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.rag.retriever import RetrievedChunk, retrieve


def _make_search_results(scores: list[float]) -> list[dict]:
    """Create mock Chroma search results with given similarity scores."""
    return [
        {
            "id": f"doc.md#chunk{i}",
            "text": f"Chunk text {i}",
            "metadata": {
                "doc_name": "doc.md",
                "doc_path": "policies/doc.md",
                "section": f"Section {i}",
                "chunk_index": i,
            },
            "similarity": score,
        }
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def mock_rag_deps():
    """Mock the embedding service and Chroma store."""
    with (
        patch("app.rag.retriever.get_embedding_service") as mock_embed,
        patch("app.rag.retriever.get_chroma_store") as mock_store,
    ):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384  # fake embedding
        mock_embed.return_value = embedder

        store = MagicMock()
        store.count.return_value = 100
        mock_store.return_value = store

        yield embedder, store


# ── Threshold filtering ──────────────────────────────────────


def test_retrieve_filters_below_threshold(mock_rag_deps):
    _, store = mock_rag_deps
    # Top score 0.8, threshold 0.4, adaptive = max(0.4, 0.8-0.2) = 0.6
    store.search.return_value = _make_search_results([0.8, 0.7, 0.5, 0.3])

    chunks = retrieve("test question", min_similarity=0.4)
    # 0.8 and 0.7 pass (>=0.6), 0.5 and 0.3 don't
    assert len(chunks) == 2
    assert chunks[0].similarity == 0.8
    assert chunks[1].similarity == 0.7


def test_retrieve_returns_empty_when_all_below_threshold(mock_rag_deps):
    _, store = mock_rag_deps
    # All scores below minimum threshold
    store.search.return_value = _make_search_results([0.3, 0.2, 0.1])

    chunks = retrieve("test question", min_similarity=0.4)
    assert chunks == []


def test_retrieve_empty_collection(mock_rag_deps):
    _, store = mock_rag_deps
    store.count.return_value = 0

    chunks = retrieve("test question")
    assert chunks == []


def test_retrieve_passes_top_k(mock_rag_deps):
    _, store = mock_rag_deps
    store.search.return_value = _make_search_results([0.9, 0.8, 0.7])

    retrieve("test question", top_k=3)
    store.search.assert_called_once()
    call_kwargs = store.search.call_args
    assert call_kwargs[1]["top_k"] == 3 or call_kwargs[0][1] == 3


def test_retrieve_returns_retrieved_chunk_objects(mock_rag_deps):
    _, store = mock_rag_deps
    store.search.return_value = _make_search_results([0.9])

    chunks = retrieve("test question")
    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].id == "doc.md#chunk0"
    assert chunks[0].similarity == 0.9
    assert "doc_name" in chunks[0].metadata


def test_adaptive_threshold_uses_top_score(mock_rag_deps):
    _, store = mock_rag_deps
    # Top score 0.55, adaptive = max(0.4, 0.55-0.2) = 0.4
    # So min threshold stays at 0.4 (not adaptive)
    store.search.return_value = _make_search_results([0.55, 0.45, 0.35])

    chunks = retrieve("test question", min_similarity=0.4)
    # 0.55 passes, 0.45 passes, 0.35 doesn't
    assert len(chunks) == 2
