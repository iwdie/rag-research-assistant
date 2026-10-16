"""The assistant: index lifecycle plus the end-to-end query path.

``RAGAssistant`` is the object the API and the CLI both talk to. It owns an
:class:`~app.services.store.IndexStore`, a :class:`~app.core.retriever.HybridRetriever`,
an optional reranker and a generator, and it keeps the "index changed" bookkeeping
in one place so the retriever is never handed a stale store.

The two construction paths are deliberately explicit:

``RAGAssistant.load(settings)``
    Load a persisted index from ``RAG_INDEX_DIR``. This is what the server does.

``RAGAssistant.build(settings, paths=...)``
    Ingest from disk and index in memory. This is what the scripts do, and what
    the server falls back to when there is no index yet and a sample corpus is
    present, so a fresh clone answers a question instead of returning 503.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.embeddings import HashingEmbedder
from app.core.filters import Filter, parse_filters
from app.core.generator import Answer, get_generator
from app.core.hybrid import RetrievedChunk
from app.core.ingest import Document, ingest_paths
from app.core.reranker import get_reranker, rerank
from app.core.retriever import HybridRetriever
from app.services.store import IndexStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Everything the API returns for one question."""

    question: str
    answer: Answer
    passages: list[RetrievedChunk]
    filters: list[Filter]
    debug: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.answer,
            "citations": [c.as_dict() for c in self.answer.citations],
            "generator": self.answer.generator,
            "used_chunks": self.answer.used_chunks,
            "passages": [p.as_dict() for p in self.passages],
            "filters": [{"field": f.field, "op": f.op, "value": f.value} for f in self.filters],
            "debug": self.debug,
        }


class RAGAssistant:
    """Hybrid retrieval over a PDF corpus, with reranking and cited answers."""

    def __init__(self, settings: Settings, store: IndexStore) -> None:
        self.settings = settings
        self.store = store
        self.reranker = (
            get_reranker(backend=settings.reranker_backend, model_name=settings.reranker_model)
            if settings.rerank
            else None
        )
        self.generator = get_generator(
            kind=settings.generator,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
        )
        self.dense_weight = self._effective_dense_weight()
        self.retriever = HybridRetriever(
            store=store,
            fusion=settings.fusion,
            rrf_k=settings.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=settings.sparse_weight,
            candidate_k=settings.candidate_k,
            top_k=settings.top_k,
            reranker=self.reranker,
            use_faiss=settings.use_faiss,
        )

    def _effective_dense_weight(self) -> float:
        """Cap the dense weight when the embedder is a lexical proxy.

        See ``Settings.lexical_dense_weight``. Doing this here rather than in the
        fusion code keeps ``reciprocal_rank_fusion`` a pure function of its input
        weights, and it is reported through ``/health`` and ``/stats`` so the
        running configuration is never a guess.
        """
        if isinstance(self.store.embedder, HashingEmbedder):
            return min(self.settings.dense_weight, self.settings.lexical_dense_weight)
        return self.settings.dense_weight

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, settings: Settings, *, paths: Iterable[Path] | None = None) -> RAGAssistant:
        store = IndexStore.from_settings(settings)
        assistant = cls(settings, store)
        if paths:
            assistant.index_paths(paths)
        return assistant

    @classmethod
    def load(cls, settings: Settings) -> RAGAssistant:
        store = IndexStore.load(
            settings.index_dir,
            embedder=IndexStore.from_settings(settings).embedder,
            use_faiss=settings.use_faiss,
        )
        return cls(settings, store)

    @classmethod
    def load_or_build(cls, settings: Settings, *, sample_dir: Path | None = None) -> RAGAssistant:
        """Load the persisted index, or bootstrap one from a sample corpus.

        Returning an assistant with an empty index is a valid state - the API
        reports ``documents: 0`` and ``/query`` explains that nothing is indexed -
        but a demo that answers nothing on first run is a bad demo, so a bundled
        sample corpus is used when one exists.
        """
        try:
            return cls.load(settings)
        except (FileNotFoundError, ValueError) as exc:
            logger.info("no usable index at %s (%s)", settings.index_dir, exc)

        paths: list[Path] = []
        if sample_dir and Path(sample_dir).is_dir():
            paths = [Path(sample_dir)]
            logger.info("bootstrapping the index from the sample corpus at %s", sample_dir)
        assistant = cls.build(settings, paths=paths)
        if paths:
            assistant.save()
        return assistant

    # -- indexing ------------------------------------------------------------
    def index_documents(self, documents: Sequence[Document]) -> int:
        """Add documents and keep the retriever pointed at the same store."""
        added = self.store.add_documents(
            documents,
            chunk_words=self.settings.chunk_words,
            overlap_words=self.settings.chunk_overlap_words,
            min_words=self.settings.min_chunk_words,
        )
        self.retriever.store = self.store
        return added

    def index_paths(self, paths: Iterable[Path], *, metadata: dict[str, Any] | None = None) -> int:
        documents = ingest_paths(paths, metadata=metadata)
        return self.index_documents(documents)

    def remove_document(self, doc_id: str) -> int:
        removed = self.store.remove_document(doc_id)
        self.retriever.store = self.store
        return removed

    def save(self) -> None:
        self.store.save(self.settings.index_dir)

    # -- querying ------------------------------------------------------------
    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: Any = None,
        rerank_results: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieval only - no answer synthesis. Used by ``POST /search`` and evaluation."""
        return self.retriever.retrieve(
            question,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=parse_filters(filters),
            rerank_results=rerank_results,
        )

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: Any = None,
        use_rerank: bool | None = None,
        debug: bool = False,
    ) -> QueryResult:
        """Retrieve, rerank and synthesise a cited answer."""
        parsed_filters = parse_filters(filters)
        settings = self.settings
        top_k = top_k or settings.top_k
        should_rerank = settings.rerank if use_rerank is None else use_rerank

        passages = self.retriever.retrieve(
            question,
            top_k=top_k,
            filters=parsed_filters,
            rerank_results=should_rerank,
        )

        # The generator reads the top of the *reranked* list, so reranking has to
        # have happened before generation. When reranking is switched off for a
        # query, the fused order is what the cross-encoder would have reordered,
        # and we say so in `debug` rather than silently changing the contract.
        context = passages[: settings.max_context_chunks]
        answer = self.generator.generate(question, context, limit=settings.max_context_chunks)

        return QueryResult(
            question=question,
            answer=answer,
            passages=passages,
            filters=parsed_filters,
            debug=self.retriever.last_debug.as_dict() if (debug and self.retriever.last_debug) else None,
        )

    def rerank_passages(self, question: str, passages: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank an externally supplied passage list (used by the eval harness)."""
        if self.reranker is None:
            return list(passages)
        return rerank(self.reranker, question, passages)

    # -- reporting -----------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.store.n_chunks else "empty",
            "documents": len(self.store.documents),
            "chunks": self.store.n_chunks,
            "embedder": self.store.embedder.name,
            "vector_backend": self.store.vectors.backend,
            "reranker": self.reranker.name if self.reranker else None,
            "generator": self.generator.name,
            "fusion": self.settings.fusion,
            "fusion_weights": {"sparse": self.settings.sparse_weight, "dense": self.dense_weight},
            "index_dir": str(self.settings.index_dir),
        }
