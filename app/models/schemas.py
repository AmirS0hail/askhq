"""Pydantic models for API requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """POST /ask request body."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask about company policies.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (overrides env default).",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (overrides env default).",
    )


# ── Responses ────────────────────────────────────────────────


class Source(BaseModel):
    """A source chunk that contributed to the answer."""

    doc_name: str
    doc_path: str
    section: str
    chunk_id: str
    similarity_score: float


class QueryResponse(BaseModel):
    """POST /ask response body."""

    answer: str
    sources: list[Source]
    query: str
    retrieved_chunks: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    ollama_connected: bool
    chroma_docs: int
    version: str = "0.1.0"


class ReindexResponse(BaseModel):
    """POST /reindex response body."""

    status: str
    documents_processed: int
    chunks_created: int
    duration_s: float
