"""Turn pages of prose into overlapping, citable chunks.

Design notes
------------
* Chunks are sentence-aligned. Cutting mid-sentence hurts both BM25 (broken
  terms) and the cross-encoder (truncated claims), and it makes quoted evidence
  read badly in the final answer.
* Overlap is expressed in *sentences*, not characters, so a fact that straddles
  a boundary still lands intact in at least one chunk.
* ``chunk_id`` is ``<doc_id>::<ordinal>``. It is stable for a given document and
  chunking config, which is what makes the labelled evaluation set reusable: an
  anchor sentence always resolves to the same chunk id.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Sentence boundary: terminator followed by whitespace and something that looks
# like the start of a new sentence. Deliberately conservative - splitting too
# eagerly is worse than missing an occasional boundary.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\s\u201d\"]+(?=[\"'(]?[A-Z0-9\u201c])")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Headings such as "3.2 Training procedure" should start a new chunk rather than
# being glued to the paragraph above them.
_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{2,80}|Abstract|Introduction|Conclusion|"
    r"References|Related Work|Method(?:s)?|Experiments?|Results?|Discussion)$"
)


def word_count(text: str) -> int:
    """Whitespace-token count. Used as a cheap, tokenizer-independent proxy."""
    return len(_WORD_RE.findall(text))


def _split_into_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def split_sentences(text: str) -> list[str]:
    """Split into sentence units, treating section headings as their own unit.

    Headings are checked per *line*, not per paragraph. ``pypdf`` returns a
    heading and the sentence under it as adjacent lines, and gluing them
    together both hides the heading from BM25 and starts the chunk with a
    fragment of the previous section.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        pending: list[str] = []
        for raw_line in block.split("\n"):
            line = " ".join(raw_line.split())
            if not line:
                continue
            if _HEADING_RE.match(line) and word_count(line) <= 12:
                if pending:
                    units.extend(_split_into_sentences(" ".join(pending)))
                    pending = []
                units.append(line)
            else:
                pending.append(line)
        if pending:
            units.extend(_split_into_sentences(" ".join(pending)))
    return units


def _hard_split(unit: str, limit: int) -> list[str]:
    """Split a single oversized sentence on word boundaries.

    Some PDFs arrive as one enormous run-on (bad extraction, tables, references
    without punctuation). Without this the chunker would happily emit a chunk
    several thousand words long and blow past the encoder's context window.
    """
    words = unit.split()
    if len(words) <= limit:
        return [unit]
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


@dataclass
class Chunk:
    """A retrievable passage plus the provenance needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    page: int | None = None
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return word_count(self.text)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "page": self.page,
            "title": self.title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=record["chunk_id"],
            doc_id=record["doc_id"],
            text=record["text"],
            ordinal=record["ordinal"],
            page=record.get("page"),
            title=record.get("title", ""),
            metadata=record.get("metadata", {}) or {},
        )


def chunk_text(
    text: str,
    *,
    doc_id: str,
    title: str = "",
    page: int | None = None,
    start_ordinal: int = 0,
    chunk_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 25,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Pack ``text`` into overlapping sentence-aligned chunks.

    ``start_ordinal`` lets a multi-page document keep one continuous numbering,
    so ``chunk_id`` stays unique across pages.
    """
    metadata = metadata or {}
    units: list[str] = []
    for unit in split_sentences(text):
        units.extend(_hard_split(unit, max(chunk_words * 2, 2)))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    ordinal = start_ordinal

    def flush() -> list[str]:
        """Emit the buffered sentences and return the overlap carried forward."""
        nonlocal current, current_words, ordinal
        if not current:
            return []
        body = " ".join(current).strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{ordinal}",
                    doc_id=doc_id,
                    text=body,
                    ordinal=ordinal,
                    page=page,
                    title=title,
                    metadata=dict(metadata),
                )
            )
            ordinal += 1
        if overlap_words <= 0:
            current, current_words = [], 0
            return []
        tail: list[str] = []
        kept = 0
        for sentence in reversed(current):
            if kept >= overlap_words:
                break
            tail.append(sentence)
            kept += word_count(sentence)
        # Never carry the whole buffer forward: that would loop on a short chunk.
        if len(tail) >= len(current):
            tail = tail[:-1] if len(current) > 1 else []
        current = list(reversed(tail))
        current_words = sum(word_count(s) for s in current)
        return current

    for unit in units:
        size = word_count(unit)
        if current and current_words + size > chunk_words:
            flush()
        current.append(unit)
        current_words += size

    flush()

    # Merge a trailing stub into its predecessor. A 3-word final chunk is
    # retrieval noise: it can never win on its own and it pollutes the metrics.
    if len(chunks) >= 2 and chunks[-1].n_words < min_words:
        tail = chunks.pop()
        chunks[-1].text = f"{chunks[-1].text} {tail.text}".strip()
    return chunks


def chunk_pages(
    pages: Iterable[str],
    *,
    doc_id: str,
    title: str = "",
    chunk_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 25,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk a page sequence, keeping page numbers for citation."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        chunks.extend(
            chunk_text(
                page_text,
                doc_id=doc_id,
                title=title,
                page=page_number,
                start_ordinal=len(chunks),
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                min_words=min_words,
                metadata=metadata,
            )
        )
    return chunks
