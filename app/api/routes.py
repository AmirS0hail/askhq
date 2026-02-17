"""API routes: /health, /ask, /reindex."""

import logging
import secrets
import time

from fastapi import APIRouter, Header, HTTPException

from app.config import get_settings
from app.models.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    Source,
)
from app.rag import rag_service
from app.rag.chunker import chunk_all_documents
from app.rag.llm_provider import get_llm_provider
from app.rag.embedder import get_embedding_service
from app.rag.vectorstore import get_chroma_store

logger = logging.getLogger(__name__)

router = APIRouter()


# ── GET /health ──────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health: checks LLM provider connectivity and Chroma doc count."""
    provider = get_llm_provider()
    llm_ok = await provider.check_health()

    try:
        store = get_chroma_store()
        doc_count = store.count()
    except Exception:
        doc_count = 0

    status = "healthy" if llm_ok and doc_count > 0 else "degraded"

    return HealthResponse(
        status=status,
        llm_connected=llm_ok,
        chroma_docs=doc_count,
    )


# ── POST /ask ────────────────────────────────────────────────


@router.post("/ask", response_model=QueryResponse)
async def ask_question(body: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline."""
    result = await rag_service.answer_question(
        question=body.question,
        top_k=body.top_k,
        min_score=body.min_score,
    )

    sources = [
        Source(
            doc_name=s["doc_name"],
            doc_path=s["doc_path"],
            section=s["section"],
            chunk_id=s["chunk_id"],
            similarity_score=s["similarity_score"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        query=result["query"],
        retrieved_chunks=result["retrieved_chunks"],
        processing_time_ms=result["processing_time_ms"],
    )


# ── POST /reindex ────────────────────────────────────────────


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(
    x_reindex_token: str = Header(..., alias="X-Reindex-Token"),
) -> ReindexResponse:
    """Rebuild the vector database from /knowledge markdown files.

    Protected by X-Reindex-Token header.
    """
    settings = get_settings()

    # Auth check - use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_reindex_token, settings.reindex_token):
        raise HTTPException(status_code=403, detail="Invalid reindex token")

    start = time.perf_counter()

    knowledge_path = settings.knowledge_path
    if not knowledge_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge directory not found: {knowledge_path}",
        )

    # Chunk all documents
    chunks = chunk_all_documents(knowledge_path)
    if not chunks:
        return ReindexResponse(
            status="warning",
            documents_processed=0,
            chunks_created=0,
            duration_s=0.0,
        )

    # Embed
    embedder = get_embedding_service()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # Store (reset collection first)
    store = get_chroma_store()
    store.delete_collection()
    chunks_created = store.add_chunks(chunks, embeddings)

    # Count unique documents
    doc_paths = {c.metadata.get("doc_path", "") for c in chunks}
    elapsed = time.perf_counter() - start

    logger.info(
        "Reindex complete: %d docs → %d chunks in %.1fs",
        len(doc_paths),
        chunks_created,
        elapsed,
    )

    return ReindexResponse(
        status="success",
        documents_processed=len(doc_paths),
        chunks_created=chunks_created,
        duration_s=round(elapsed, 2),
    )
