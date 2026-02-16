"""Markdown document chunker.

Parses markdown files with optional YAML frontmatter and splits them into
chunks by heading boundaries. Long sections are further split by paragraphs.

Chunk size is capped at ~250 tokens to stay within the all-MiniLM-L6-v2
embedding model's 256 wordpiece-token limit.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

logger = logging.getLogger(__name__)

# Approximate: 1 word ≈ 1.3 tokens for English text.
MAX_CHUNK_WORDS = 180  # ≈ 230 tokens — safe margin under 256

# Regex: split on lines that start with ## or deeper headings
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)", re.MULTILINE)


@dataclass
class Chunk:
    """A single text chunk with metadata for indexing."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _word_count(text: str) -> int:
    return len(text.split())


def _doc_slug(path: Path) -> str:
    """Create deterministic slug from file path relative to knowledge dir."""
    return path.stem.lower().replace(" ", "-")


def _make_chunk_id(doc_path: str, index: int) -> str:
    """Deterministic chunk ID: <doc_path>#chunk<index>."""
    return f"{doc_path}#chunk{index}"


def _split_paragraphs_with_overlap(text: str, max_words: int) -> list[str]:
    """Split text by paragraphs, accumulating until word limit, with 1-sentence overlap.
    
    Paragraphs are accumulated until adding the next one would exceed max_words.
    When splitting, the last sentence of the previous chunk is carried over as overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        pw = _word_count(para)
        if current_words + pw > max_words and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # Overlap: carry last sentence of previous chunk using robust sentence split
            # Split on sentence boundaries (., !, ?) followed by whitespace
            sentences = re.split(r'(?<=[.!?])\s+', current[-1])
            # Always take at least the last sentence; if only one sentence, take the whole para
            overlap = sentences[-1] if sentences else current[-1]
            current = [overlap]
            current_words = _word_count(overlap)
        current.append(para)
        current_words += pw

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _extract_sections(body: str) -> list[tuple[list[str], str]]:
    """Split markdown body into (heading_path, content) tuples.

    Returns a list of (section_path, section_text) where section_path is the
    hierarchy of headings leading to this section (e.g. ["Types of Leave",
    "Casual Leave"]).

    Content before any heading is returned with an empty path.
    """
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    # Split body into segments at heading boundaries
    parts = _HEADING_RE.split(body)

    # parts[0] is content before first heading
    pre_heading = parts[0].strip()
    if pre_heading:
        sections.append(([], pre_heading))

    # After that, parts come in triples: (hashes, title, content)
    i = 1
    while i < len(parts):
        level = len(parts[i])            # number of # characters
        title = parts[i + 1].strip()
        content = parts[i + 2].strip() if i + 2 < len(parts) else ""
        i += 3

        # Update heading stack: pop headings at same or deeper level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        path = [h[1] for h in heading_stack]

        if content:
            sections.append((path, content))

    return sections


def chunk_document(
    file_path: Path,
    knowledge_dir: Path,
    max_words: int = MAX_CHUNK_WORDS,
) -> list[Chunk]:
    """Parse a markdown file and return a list of Chunks.

    Args:
        file_path: Absolute path to the markdown file.
        knowledge_dir: Root knowledge directory (for relative path computation).
        max_words: Maximum words per chunk.

    Returns:
        List of Chunk objects with deterministic IDs and metadata.
    """
    raw = file_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    post = frontmatter.loads(raw)
    fm: dict[str, Any] = dict(post.metadata)
    body: str = post.content

    # Relative path for metadata
    try:
        rel_path = str(file_path.relative_to(knowledge_dir))
    except ValueError:
        rel_path = file_path.name

    doc_name = file_path.name
    slug = _doc_slug(file_path)

    # Extract H1 title if present
    h1_match = re.match(r"^#\s+(.*)", body, re.MULTILINE)
    doc_title = fm.get("title", h1_match.group(1) if h1_match else doc_name)

    # Remove H1 from body (it's document-level, not a chunk section)
    body_no_h1 = re.sub(r"^#\s+.*\n?", "", body, count=1).strip()

    # Split into sections by headings
    sections = _extract_sections(body_no_h1)

    chunks: list[Chunk] = []
    chunk_index = 0

    for section_path, section_text in sections:
        heading_label = " > ".join([doc_title] + section_path) if section_path else doc_title
        section_title = section_path[-1] if section_path else doc_title

        # Split long sections by paragraphs
        if _word_count(section_text) > max_words:
            sub_texts = _split_paragraphs_with_overlap(section_text, max_words)
        else:
            sub_texts = [section_text]

        for text in sub_texts:
            if not text.strip():
                continue

            chunk_id = _make_chunk_id(rel_path, chunk_index)

            # Build metadata — flat primitives only (Chroma requirement)
            meta: dict[str, Any] = {
                "doc_name": doc_name,
                "doc_path": rel_path,
                "doc_slug": slug,
                "doc_title": doc_title,
                "section": heading_label,
                "chunk_title": section_title,
                "chunk_index": chunk_index,
                "source_type": "markdown",
                "word_count": _word_count(text),
            }

            # Add frontmatter fields (flatten to strings)
            for key, value in fm.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[f"fm_{key}"] = value
                elif isinstance(value, list):
                    meta[f"fm_{key}"] = ", ".join(str(v) for v in value)

            chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))
            chunk_index += 1

    # Backfill total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.metadata["total_chunks"] = total

    logger.info("Chunked %s → %d chunks", rel_path, total)
    return chunks


def chunk_all_documents(knowledge_dir: Path, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    """Walk a knowledge directory and chunk all .md files.

    Args:
        knowledge_dir: Root directory to scan recursively.
        max_words: Maximum words per chunk.

    Returns:
        Combined list of Chunk objects from all documents.
    """
    all_chunks: list[Chunk] = []
    md_files = sorted(knowledge_dir.rglob("*.md"))

    if not md_files:
        logger.warning("No .md files found in %s", knowledge_dir)
        return all_chunks

    for md_file in md_files:
        try:
            doc_chunks = chunk_document(md_file, knowledge_dir, max_words)
            all_chunks.extend(doc_chunks)
        except Exception:
            logger.exception("Failed to chunk %s", md_file)

    logger.info(
        "Total: %d documents → %d chunks",
        len(md_files),
        len(all_chunks),
    )
    return all_chunks
