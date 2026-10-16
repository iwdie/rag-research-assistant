"""Index persistence and the query-facing view of the corpus.

An :class:`IndexStore` owns the three things that must agree with each other:
the chunk records, the embedding matrix, and the BM25 postings. Keeping them
behind one object means there is exactly one place that can put them out of sync
(``add_documents``), and one place that reloads them (``load``).

On disk::

    data/index/
      meta.json          config fingerprint + counts + backend names
      documents.jsonl    one record per source document
      chunks.jsonl       one record per chunk (text, page, metadata)
      vectors.npy        (n, dim) float32, row i == chunks.jsonl line i
      vector_ids.json    chunk ids in matrix row order

The BM25 index is rebuilt from the chunk texts on load rather than persisted.
It is a pure function of the texts and costs well under a second for thousands
of chunks - far less than shipping a serialised postings file that can disagree
with ``chunks.jsonl``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.chunking import Chunk
from app.core.embeddings import Embedder, get_embedder
from app.core.ingest import Document, chunk_documents
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
DOCUMENTS_FILE = "documents.jsonl"
CHUNKS_FILE = "chunks.jsonl"


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class IndexStore:
    """Chunks, embeddings and BM25 postings for one corpus."""

    embedder: Embedder
    use_faiss: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    bm25: BM25Index = field(default_factory=BM25Index)
    vectors: VectorStore | None = None
    _by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.vectors is None:
            self.vectors = VectorStore(self.embedder.dim, use_faiss=self.use_faiss)
        self.bm25 = BM25Index(k1=self.bm25_k1, b=self.bm25_b)
        self._by_id = {}

    # -- construction --------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> IndexStore:
        embedder = get_embedder(
            backend=settings.embedder_backend,
            model_name=settings.embedding_model,
            hashing_dim=settings.hashing_dim,
            batch_size=settings.embedding_batch_size,
        )
        return cls(
            embedder=embedder,
            use_faiss=settings.use_faiss,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
        )

    # -- access --------------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def filter_view(self, index: int, chunk: Chunk) -> dict[str, Any]:
        """The metadata a filter sees for a chunk.

        Top-level chunk fields and the document metadata dict are merged so a
        filter can say ``doc_id=a1b2`` (structural) or ``year=2017`` (semantic)
        without the caller knowing which layer stores it.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "page": chunk.page,
            "title": chunk.title,
            "ordinal": chunk.ordinal,
            "index": index,
            **chunk.metadata,
        }

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def search_dense(
        self,
        query: str,
        *,
        k: int = 10,
        allowed_indices: Sequence[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Embed the query and search the vector store."""
        if not query.strip() or self.n_chunks == 0:
            return []
        vector = self.embedder.encode([query])[0]
        return self.vectors.search(vector, k=k, allowed_indices=allowed_indices)

    # -- mutation ------------------------------------------------------------
    def add_documents(self, documents: Sequence[Document], *, chunk_words: int, overlap_words: int, min_words: int) -> int:
        """Chunk, embed and index new documents. Returns the number of new chunks.

        Documents already present (same content-addressed ``doc_id``) are
        replaced rather than duplicated, so re-running ingestion is idempotent.
        """
        new_documents = [document for document in documents if document.doc_id not in self.documents]
        if not new_documents:
            return 0

        chunks = chunk_documents(
            new_documents,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
        )
        if not chunks:
            self.documents.update({document.doc_id: document for document in new_documents})
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.encode(texts)
        self.vectors.add([chunk.chunk_id for chunk in chunks], vectors)
        self.bm25.extend([chunk.chunk_id for chunk in chunks], texts)
        for chunk in chunks:
            self.chunks.append(chunk)
            self._by_id[chunk.chunk_id] = chunk
        self.documents.update({document.doc_id: document for document in new_documents})
        logger.info("indexed %d chunks from %d documents", len(chunks), len(new_documents))
        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Drop a document and rebuild the derived structures.

        A removal has to rebuild both indexes: FAISS has no cheap delete from an
        ``IndexFlat``, and BM25's IDF changes when the corpus size changes.
        """
        if doc_id not in self.documents:
            return 0
        kept_chunks = [chunk for chunk in self.chunks if chunk.doc_id != doc_id]
        removed = len(self.chunks) - len(kept_chunks)
        if kept_chunks:
            vectors = self.embedder.encode([chunk.text for chunk in kept_chunks])
        else:
            vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)

        self.vectors.clear()
        self.vectors.add([chunk.chunk_id for chunk in kept_chunks], vectors)
        self.bm25 = BM25Index(k1=self.bm25_k1, b=self.bm25_b).fit(
            [chunk.chunk_id for chunk in kept_chunks], [chunk.text for chunk in kept_chunks]
        )
        self.chunks = kept_chunks
        self._by_id = {chunk.chunk_id: chunk for chunk in kept_chunks}
        self.documents.pop(doc_id, None)
        return removed

    def rebuild(self) -> None:
        """Recompute every derived structure from the chunk list."""
        texts = [chunk.text for chunk in self.chunks]
        self.vectors.clear()
        if texts:
            self.vectors.add([chunk.chunk_id for chunk in self.chunks], self.embedder.encode(texts))
        self.bm25 = BM25Index(k1=self.bm25_k1, b=self.bm25_b).fit(
            [chunk.chunk_id for chunk in self.chunks], texts
        )
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors.save(directory)
        _write_jsonl(directory / CHUNKS_FILE, (chunk.to_record() for chunk in self.chunks))
        _write_jsonl(directory / DOCUMENTS_FILE, (doc.to_record() for doc in self.documents.values()))
        meta = {
            "version": 1,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "vector_backend": self.vectors.backend,
            "bm25": {"k1": self.bm25_k1, "b": self.bm25_b},
            "n_documents": len(self.documents),
            "n_chunks": self.n_chunks,
            "sources": sorted(doc.source for doc in self.documents.values()),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("saved index (%d chunks) to %s", self.n_chunks, directory)

    @classmethod
    def load(cls, directory: Path, *, embedder: Embedder, use_faiss: bool = True) -> IndexStore:
        """Load a persisted index. Raises if the directory holds nothing usable."""
        directory = Path(directory)
        if not (directory / META_FILE).exists():
            raise FileNotFoundError(f"no index at {directory}: {META_FILE} is missing")
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if meta.get("dim") != embedder.dim:
            raise ValueError(
                f"index was built with {meta.get('embedder')} (dim {meta.get('dim')}) but the "
                f"configured embedder {embedder.name} produces dim {embedder.dim}. "
                "Set RAG_EMBEDDER_BACKEND/RAG_EMBEDDING_MODEL to match, or re-index."
            )
        store = cls(
            embedder=embedder,
            use_faiss=use_faiss,
            bm25_k1=float(meta.get("bm25", {}).get("k1", 1.5)),
            bm25_b=float(meta.get("bm25", {}).get("b", 0.75)),
        )
        store.chunks = [Chunk.from_record(record) for record in _read_jsonl(directory / CHUNKS_FILE)]
        store._by_id = {chunk.chunk_id: chunk for chunk in store.chunks}
        store.vectors.load(directory)

        for record in _read_jsonl(directory / DOCUMENTS_FILE):
            store.documents[record["doc_id"]] = Document(
                doc_id=record["doc_id"],
                title=record.get("title", ""),
                source=record.get("source", ""),
                # Page text is not persisted: it is only needed at ingestion time,
                # and keeping it would triple the index size for no query benefit.
                pages=[],
                metadata=record.get("metadata", {}) or {},
            )
        store.bm25.fit([chunk.chunk_id for chunk in store.chunks], [chunk.text for chunk in store.chunks])
        logger.info("loaded index from %s (%d chunks)", directory, store.n_chunks)
        return store

    # -- reporting -----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        by_doc: dict[str, int] = {}
        for chunk in self.chunks:
            by_doc[chunk.doc_id] = by_doc.get(chunk.doc_id, 0) + 1
        return {
            "n_documents": len(self.documents),
            "n_chunks": self.n_chunks,
            "chunks_per_document": by_doc,
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "vector_backend": self.vectors.backend,
            "bm25": {"k1": self.bm25_k1, "b": self.bm25_b, "vocabulary": self.bm25.vocabulary},
            "total_words": sum(len(chunk.text.split()) for chunk in self.chunks),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        return [doc.to_record() | {"chunk_count": sum(1 for c in self.chunks if c.doc_id == doc.doc_id)}
                for doc in sorted(self.documents.values(), key=lambda d: d.title)]
