"""Chroma vector store wrapper.

Uses PersistentClient for durable storage. Handles the cosine distance →
similarity conversion and Chroma's flat-metadata-only constraint.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from app.config import get_settings
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper around a Chroma persistent collection."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection '%s' ready (%d documents)",
            collection_name,
            self._collection.count(),
        )

    # ── Write ────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert chunks with pre-computed embeddings into Chroma.

        Args:
            chunks: List of Chunk objects.
            embeddings: numpy array of shape (len(chunks), dim).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        # Chroma upsert in batches of 500
        batch_size = 500
        total = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeds = embeddings[i : i + batch_size]

            self._collection.upsert(
                ids=[c.id for c in batch_chunks],
                documents=[c.text for c in batch_chunks],
                embeddings=[e.tolist() for e in batch_embeds],
                metadatas=[_sanitize_metadata(c.metadata) for c in batch_chunks],
            )
            total += len(batch_chunks)

        logger.info("Upserted %d chunks into Chroma", total)
        return total

    def delete_collection(self) -> None:
        """Delete and recreate the collection (for full re-index)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset", self._collection_name)

    # ── Read ─────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query collection and return results with similarity scores.

        Chroma cosine space returns distance = 1 - similarity.
        We convert to similarity for downstream thresholding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of dicts with keys: id, text, metadata, similarity.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return items

        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]  # type: ignore[index]
            similarity = 1.0 - distance  # cosine distance → similarity

            items.append(
                {
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],  # type: ignore[index]
                    "metadata": results["metadatas"][0][idx],  # type: ignore[index]
                    "similarity": round(similarity, 4),
                }
            )

        return items

    def count(self) -> int:
        """Return total documents in collection."""
        return self._collection.count()


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten metadata to Chroma-compatible primitive types.

    Chroma only accepts str, int, float, bool as metadata values.
    Lists and other types are coerced to strings.
    """
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            clean[key] = ""
        else:
            clean[key] = str(value)
    return clean


@lru_cache(maxsize=1)
def get_chroma_store() -> ChromaStore:
    """Return cached ChromaStore singleton."""
    settings = get_settings()
    return ChromaStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
