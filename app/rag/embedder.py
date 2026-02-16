"""Embedding service using SentenceTransformers.

Lazy-loaded singleton to avoid model reload on FastAPI hot-reload.
Uses all-MiniLM-L6-v2 (384-dim, 256 wordpiece token limit).
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around SentenceTransformer for document/query embedding."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query text.

        Args:
            text: Query string to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        return self._model.encode(
            text,
            normalize_embeddings=True,
        )


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return cached EmbeddingService singleton."""
    settings = get_settings()
    return EmbeddingService(settings.embedding_model)
