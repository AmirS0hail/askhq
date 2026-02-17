"""RAG orchestration service.

Coordinates retrieval → prompt assembly → LLM generation.
Handles fallback cases: no relevant chunks, Ollama unavailable.
"""

import logging
import time
from typing import Any

from app.rag.llm_provider import LLMError, get_llm_provider
from app.rag.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────

SYSTEM_PROMPT = """You are AskHQ, Alfabolt's internal HR policy assistant.

STRICT RULES:
1. Answer ONLY using the policy excerpts provided below.
2. If the answer is NOT in the provided context, say: "I couldn't find this in the current policy documents. Please check with HR."
3. NEVER add information from general knowledge or make assumptions.
4. Be concise — use bullet points where appropriate.
5. Cite the source document and section at the end of your answer.

Policy Context:
{context}

Question: {question}

Answer:"""

NO_CONTEXT_RESPONSE = (
    "I couldn't find this in the current policy documents. "
    "Please check with Alfabolt HR."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.metadata.get("section", "Unknown")
        doc = chunk.metadata.get("doc_name", "Unknown")
        parts.append(f"[Excerpt {i} — {doc}: {section}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Extract source metadata from chunks for the API response."""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        # Deduplicate by chunk ID
        if chunk.id in seen:
            continue
        seen.add(chunk.id)

        sources.append(
            {
                "doc_name": chunk.metadata.get("doc_name", ""),
                "doc_path": chunk.metadata.get("doc_path", ""),
                "section": chunk.metadata.get("section", ""),
                "chunk_id": chunk.id,
                "similarity_score": chunk.similarity,
            }
        )

    return sources


async def answer_question(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """End-to-end: question → retrieval → LLM → answer.

    Args:
        question: User question string.
        top_k: Override top-K retrieval count.
        min_score: Override minimum similarity threshold.

    Returns:
        Dict with keys: answer, sources, query, retrieved_chunks,
        processing_time_ms.
    """
    start = time.perf_counter()

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question, top_k=top_k, min_similarity=min_score)

    # Step 2: No relevant chunks → return fallback WITHOUT calling LLM
    if not chunks:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "No chunks above threshold for: '%s' (%.0fms)",
            question[:60],
            elapsed_ms,
        )
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "sources": [],
            "query": question,
            "retrieved_chunks": 0,
            "processing_time_ms": round(elapsed_ms, 1),
        }

    # Step 3: Build prompt with context
    context = _format_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    # Step 4: Call LLM (with fallback if provider is down)
    try:
        provider = get_llm_provider()
        answer = await provider.generate(prompt)
    except LLMError as e:
        logger.warning("LLM unavailable, returning raw excerpts: %s", e)
        # Graceful degradation: return raw chunk content
        excerpts = "\n\n".join(
            f"• **{c.metadata.get('section', 'Policy')}**: {c.text[:300]}"
            for c in chunks[:3]
        )
        answer = (
            "⚠️ AI summarization is temporarily unavailable. "
            "Here are the relevant policy excerpts:\n\n" + excerpts
        )

    sources = _build_sources(chunks)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Answered: '%s' → %d sources, %.0fms",
        question[:60],
        len(sources),
        elapsed_ms,
    )

    return {
        "answer": answer,
        "sources": sources,
        "query": question,
        "retrieved_chunks": len(chunks),
        "processing_time_ms": round(elapsed_ms, 1),
    }
