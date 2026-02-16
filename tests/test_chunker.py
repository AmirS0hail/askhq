"""Tests for the markdown chunker."""

from pathlib import Path

import pytest

from app.rag.chunker import Chunk, chunk_document, _extract_sections, _split_paragraphs_with_overlap


# ── Fixtures ─────────────────────────────────────────────────

SAMPLE_MD = """\
---
title: Test Policy
category: testing
version: "1.0"
---

# Test Policy Document

Introduction paragraph before any section.

## Section One

Content of section one with some details.

### Subsection A

Details about subsection A.

- Bullet point one
- Bullet point two

### Subsection B

Details about subsection B.

## Section Two

Content of section two.

More content in section two with multiple paragraphs.

Yet another paragraph in section two.
"""


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Write sample markdown to a temp file."""
    md_file = tmp_path / "test-policy.md"
    md_file.write_text(SAMPLE_MD, encoding="utf-8")
    return md_file


# ── Section extraction ───────────────────────────────────────


def test_extract_sections_splits_by_headings():
    body = """\
Intro text.

## First Section

First content.

### Nested

Nested content.

## Second Section

Second content.
"""
    sections = _extract_sections(body)
    assert len(sections) >= 3

    # First item is intro (no heading path)
    assert sections[0][0] == []
    assert "Intro text" in sections[0][1]

    # Second is "First Section" content
    assert sections[1][0] == ["First Section"]

    # Third is nested under First Section
    assert sections[2][0] == ["First Section", "Nested"]


def test_extract_sections_empty_body():
    sections = _extract_sections("")
    assert sections == []


# ── Paragraph splitting ──────────────────────────────────────


def test_split_paragraphs_short_text():
    text = "Short paragraph."
    result = _split_paragraphs_with_overlap(text, max_words=100)
    assert len(result) == 1
    assert result[0] == text


def test_split_paragraphs_splits_long_text():
    # Create text with multiple paragraphs exceeding max_words
    para = "Word " * 50  # 50 words per paragraph
    text = f"{para.strip()}\n\n{para.strip()}\n\n{para.strip()}"
    result = _split_paragraphs_with_overlap(text, max_words=80)
    assert len(result) >= 2


# ── Full chunking ────────────────────────────────────────────


def test_chunk_document_produces_chunks(sample_md_file: Path):
    chunks = chunk_document(sample_md_file, sample_md_file.parent)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_document_has_metadata(sample_md_file: Path):
    chunks = chunk_document(sample_md_file, sample_md_file.parent)

    for chunk in chunks:
        # Required metadata fields
        assert "doc_name" in chunk.metadata
        assert "doc_path" in chunk.metadata
        assert "section" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata
        assert "source_type" in chunk.metadata
        assert chunk.metadata["source_type"] == "markdown"


def test_chunk_document_preserves_frontmatter(sample_md_file: Path):
    chunks = chunk_document(sample_md_file, sample_md_file.parent)
    assert len(chunks) > 0

    # Frontmatter fields should be prefixed with fm_
    first = chunks[0]
    assert first.metadata.get("fm_title") == "Test Policy"
    assert first.metadata.get("fm_category") == "testing"


def test_chunk_ids_are_deterministic(sample_md_file: Path):
    chunks_a = chunk_document(sample_md_file, sample_md_file.parent)
    chunks_b = chunk_document(sample_md_file, sample_md_file.parent)
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]


def test_chunk_ids_are_unique(sample_md_file: Path):
    chunks = chunk_document(sample_md_file, sample_md_file.parent)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_text_not_empty(sample_md_file: Path):
    chunks = chunk_document(sample_md_file, sample_md_file.parent)
    for chunk in chunks:
        assert chunk.text.strip(), f"Chunk {chunk.id} has empty text"


def test_leave_policy_chunking():
    """Test against the actual sample leave policy document."""
    leave_policy = Path("knowledge/policies/leave-policy.md")
    if not leave_policy.exists():
        pytest.skip("Leave policy not found at expected path")

    knowledge_dir = Path("knowledge")
    chunks = chunk_document(leave_policy, knowledge_dir)

    assert len(chunks) >= 5, f"Expected >=5 chunks, got {len(chunks)}"

    # Check that key sections are captured
    all_text = " ".join(c.text for c in chunks)
    assert "3 days per year" in all_text  # casual leave
    assert "7 days per year" in all_text  # sick leave
    assert "14 days per year" in all_text  # annual leave
    assert "3 months" in all_text  # maternity leave
