"""Retrieval service: embed query → search Chroma → threshold filter.

Converts Chroma distances to similarities and applies adaptive thresholding
to reject low-confidence results.
"""

import logging
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.rag.embedder import get_embedding_service
from app.rag.vectorstore import get_chroma_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its similarity score."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float


def retrieve(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks for a query.

    Steps:
        1. Embed the query.
        2. Search Chroma for top_k nearest chunks.
        3. Convert distances to similarity scores.
        4. Apply adaptive threshold: max(min_similarity, top_score - 0.2).
        5. Return filtered, ranked results.

    Args:
        query: User question string.
        top_k: Override for number of results (default from settings).
        min_similarity: Override for threshold (default from settings).

    Returns:
        List of RetrievedChunk ordered by similarity (highest first).
        Empty list if nothing passes threshold.
    """
    settings = get_settings()
    k = top_k or settings.top_k
    threshold = min_similarity if min_similarity is not None else settings.min_similarity

    # Embed query
    embedder = get_embedding_service()
    query_embedding = embedder.embed_query(query)

    # Search
    store = get_chroma_store()
    if store.count() == 0:
        logger.warning("Chroma collection is empty — no documents indexed")
        return []

    results = store.search(query_embedding, top_k=k)

    if not results:
        logger.info("No results returned from Chroma for query: %s", query[:80])
        return []

    # Adaptive threshold: at least min_similarity, but also within 0.2 of top hit
    top_score = results[0]["similarity"]
    adaptive_threshold = max(threshold, top_score - 0.2)

    logger.info(
        "Query: '%s' | top_score=%.3f | threshold=%.3f (adaptive=%.3f)",
        query[:60],
        top_score,
        threshold,
        adaptive_threshold,
    )

    # Filter
    chunks: list[RetrievedChunk] = []
    for r in results:
        if r["similarity"] >= adaptive_threshold:
            chunks.append(
                RetrievedChunk(
                    id=r["id"],
                    text=r["text"],
                    metadata=r["metadata"],
                    similarity=r["similarity"],
                )
            )
            logger.debug(
                "  ✓ chunk=%s score=%.3f section=%s",
                r["id"],
                r["similarity"],
                r["metadata"].get("section", "?"),
            )
        else:
            logger.debug(
                "  ✗ chunk=%s score=%.3f (below %.3f)",
                r["id"],
                r["similarity"],
                adaptive_threshold,
            )

    logger.info(
        "Retrieved %d/%d chunks above threshold %.3f",
        len(chunks),
        len(results),
        adaptive_threshold,
    )
    return chunks
