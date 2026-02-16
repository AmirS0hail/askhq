"""CLI script to rebuild the Chroma vector database from /knowledge.

Usage:
    python -m scripts.reindex [--clean]

Flags:
    --clean   Delete existing collection before indexing (default: upsert).
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.chunker import chunk_all_documents
from app.rag.embedder import get_embedding_service
from app.rag.vectorstore import get_chroma_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reindex")


def write_chunks_jsonl(chunks, output_path: Path) -> None:
    """Write chunks to JSONL for debugging / inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "text": chunk.text[:500],  # truncate for readability
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), output_path)


def main() -> None:
    clean = "--clean" in sys.argv
    settings = get_settings()
    knowledge_path = settings.knowledge_path

    logger.info("=" * 60)
    logger.info("AskHQ Reindex")
    logger.info("Knowledge dir: %s", knowledge_path)
    logger.info("Chroma dir:    %s", settings.chroma_dir)
    logger.info("Clean mode:    %s", clean)
    logger.info("=" * 60)

    if not knowledge_path.exists():
        logger.error("Knowledge directory does not exist: %s", knowledge_path)
        sys.exit(1)

    start = time.perf_counter()

    # Step 1: Chunk all documents
    logger.info("Step 1/3: Chunking documents...")
    chunks = chunk_all_documents(knowledge_path)

    if not chunks:
        logger.warning("No chunks produced. Add .md files to %s", knowledge_path)
        sys.exit(0)

    # Write debug JSONL
    jsonl_path = Path("data/chunks.jsonl")
    write_chunks_jsonl(chunks, jsonl_path)

    # Step 2: Embed
    logger.info("Step 2/3: Generating embeddings...")
    embedder = get_embedding_service()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)
    logger.info("Embedded %d chunks (dim=%d)", len(texts), embeddings.shape[1])

    # Step 3: Store in Chroma
    logger.info("Step 3/3: Storing in Chroma...")
    store = get_chroma_store()

    if clean:
        logger.info("Cleaning existing collection...")
        store.delete_collection()

    count = store.add_chunks(chunks, embeddings)

    elapsed = time.perf_counter() - start

    # Count unique documents
    doc_paths = {c.metadata.get("doc_path", "") for c in chunks}

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Documents: %d", len(doc_paths))
    logger.info("  Chunks:    %d", count)
    logger.info("  Time:      %.1fs", elapsed)
    logger.info("  Debug:     %s", jsonl_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
