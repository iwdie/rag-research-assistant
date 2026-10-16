"""Corpus loading: PDFs, plain text and metadata sidecars.

PDF text comes from ``pypdf``, page by page, because page numbers are what make
a citation checkable by a human. Extraction is imperfect - ligatures, hyphenated
line breaks, inline references - so a little normalisation before chunking pays
for itself in retrieval quality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.chunking import Chunk, chunk_pages

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# "…trans-\nformer…" is one word broken across a line; rejoin before chunking so
# BM25 sees "transformer" and not two non-terms.
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
# A 4-digit year delimited by non-digits. Deliberately not r"\b(19|20)\d{2}\b":
# "_" is a word character, so "vaswani_2017_attention" has no word boundary
# before the year and the pattern would silently never match the most common
# filename convention there is.
_YEAR_RE = re.compile(r"(?:^|[^0-9])((?:19|20)\d{2})(?:[^0-9]|$)")


@dataclass
class Document:
    """A source document and its extracted pages."""

    doc_id: str
    title: str
    source: str
    pages: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def to_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "n_pages": self.n_pages,
            "n_words": len(self.text.split()),
            "metadata": self.metadata,
        }


def normalise_text(text: str) -> str:
    """Repair the artefacts that PDF extraction leaves behind."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse runs of blank lines to a single paragraph break: the chunker uses
    # blank lines as paragraph boundaries and repeated them as extra splits.
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            collapsed.append(line)
            blank = False
        elif not blank:
            collapsed.append("")
            blank = True
    return "\n".join(collapsed).strip()


def _doc_id(source: str, content: str) -> str:
    """Stable id from the source name and the first bytes of content.

    Content-addressed so re-ingesting the same file is a no-op, and so two files
    with the same name in different directories do not collide. Only the head is
    hashed to keep this cheap on a 300-page monograph.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(Path(source).name.encode("utf-8"))
    digest.update(content[:8192].encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


def _guess_title(path: Path) -> str:
    """Filename as a title: drop the extension, turn separators into spaces."""
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(word.capitalize() if word.islower() else word for word in stem.split())


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    """Read ``<file>.meta.json`` beside the document, if present."""
    for candidate in (path.with_suffix(path.suffix + ".meta.json"), path.with_suffix(".meta.json")):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("ignoring malformed sidecar %s: %s", candidate.name, exc)
    return {}


def load_pdf(path: Path) -> list[str]:
    """Extract per-page text from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(normalise_text(page.extract_text() or ""))
        except Exception as exc:
            logger.warning("page %d of %s failed to extract: %s", number, path.name, exc)
            pages.append("")
    return pages


def load_text(path: Path) -> list[str]:
    """Load a text or markdown file, split on a ``---`` or ``# Page`` marker."""
    raw = normalise_text(path.read_text(encoding="utf-8", errors="ignore"))
    if "---" in raw:
        parts = [part.strip() for part in raw.split("\n---\n") if part.strip()]
        if len(parts) > 1:
            return parts
    return [raw] if raw else []


def load_document(path: Path, *, metadata: dict[str, Any] | None = None) -> Document:
    """Load one file into a :class:`Document`."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        pages = load_text(path)
    else:
        raise ValueError(f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}")

    merged = _sidecar_metadata(path)
    merged.update(metadata or {})
    if not pages or not any(page.strip() for page in pages):
        logger.warning("%s produced no extractable text (scanned image?)", path.name)

    title = merged.pop("title", None) or _guess_title(path)
    body = "\n\n".join(pages)
    merged.setdefault("source_name", path.name)
    # A year in the filename is a common and reliable metadata signal
    # ("vaswani_2017_attention.pdf"); harvest it when nothing better exists.
    if "year" not in merged:
        match = _YEAR_RE.search(path.stem)
        if match:
            merged["year"] = int(match.group(1))

    metadata_clean = {**merged, "pages": len(pages)}
    return Document(
        doc_id=_doc_id(str(path), body),
        title=title,
        source=str(path),
        pages=pages,
        metadata=metadata_clean,
    )


def ingest_paths(paths: Iterable[Path], *, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Load every supported file in ``paths`` (directories are walked)."""
    documents: list[Document] = []
    for path in expand_paths(paths):
        try:
            document = load_document(path, metadata=metadata)
        except Exception as exc:
            logger.error("failed to ingest %s: %s", path, exc)
            continue
        if document.pages:
            documents.append(document)
    return documents


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    """Resolve files and directories into a sorted list of supported files."""
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file())
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            resolved.append(path)
        else:
            logger.warning("skipping %s: not a supported file or directory", raw)
    return resolved


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 25,
) -> list[Chunk]:
    """Chunk a document collection, propagating document metadata onto chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_pages(
                document.pages,
                doc_id=document.doc_id,
                title=document.title,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                min_words=min_words,
                metadata={
                    **document.metadata,
                    "doc_title": document.title,
                    "source": document.source,
                },
            )
        )
    return chunks
