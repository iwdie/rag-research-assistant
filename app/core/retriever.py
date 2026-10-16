"""The hybrid retriever: filter -> BM25 + dense -> fuse -> rerank.

This is the one place that knows how all the pieces fit together. It takes an
:class:`~app.services.store.IndexStore` (chunks, embeddings, BM25) and returns
ordered :class:`RetrievedChunk` objects ready to be cited.

Flow for a query:

1. Resolve the metadata filter once, into a list of allowed row indices. Both
   stores accept that list, so a filtered query and an unfiltered one run the
   same code with the same guarantees.
2. Ask BM25 and the vector store for ``candidate_k`` results each.
3. Fuse the two rankings (RRF by default) into a single candidate pool.
4. Optionally rerank the head of that pool. Reranking is applied to the pool,
   not to the final ``top_k``, so the cross-encoder actually gets to change the
   answer rather than shuffle five already-chosen passages.
5. Attach the full chunk record (text, title, page, metadata) to each hit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.filters import Filter, matches_all
from app.core.hybrid import RankedList, RetrievedChunk, fuse
from app.core.reranker import Reranker, rerank

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDebug:
    """Per-stage diagnostics, surfaced by ``POST /search?debug=true``."""

    filtered_rows: int
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int
    top_k: int
    fusion: str
    reranker: str | None
    rerank_applied: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "filtered_rows": self.filtered_rows,
            "dense_candidates": self.dense_candidates,
            "sparse_candidates": self.sparse_candidates,
            "fused_candidates": self.fused_candidates,
            "top_k": self.top_k,
            "fusion": self.fusion,
            "reranker": self.reranker,
            "rerank_applied": self.rerank_applied,
        }


def _ranks(results: Sequence[tuple[str, float]]) -> dict[str, int]:
    """Map id -> 1-based rank for one retriever's list."""
    return {chunk_id: position for position, (chunk_id, _) in enumerate(results, start=1)}


class HybridRetriever:
    """BM25 + dense retrieval with reciprocal rank fusion and optional reranking."""

    def __init__(
        self,
        *,
        store: Any,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_k: int = 50,
        top_k: int = 5,
        reranker: Reranker | None = None,
        use_faiss: bool = True,
    ) -> None:
        self.store = store
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.reranker = reranker
        self.use_faiss = use_faiss
        self.last_debug: RetrievalDebug | None = None

    # -- helpers -------------------------------------------------------------
    def allowed_rows(self, filters: Sequence[Filter]) -> list[int] | None:
        """Row indices permitted by ``filters``; ``None`` means unrestricted.

        Public because the evaluation harness needs to run the single-retriever
        baselines over exactly the same candidate set as the hybrid pipeline.
        """
        if not filters:
            return None
        rows = [
            index
            for index, chunk in enumerate(self.store.chunks)
            if matches_all(filters, self.store.filter_view(index, chunk))
        ]
        logger.debug("metadata filter kept %d/%d chunks", len(rows), len(self.store.chunks))
        return rows

    def _materialise(
        self,
        ranked: Sequence[tuple[str, float]],
        sparse_ranks: dict[str, int],
        dense_ranks: dict[str, int],
        scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        """Turn fused ids into ``RetrievedChunk`` objects, preserving fused order."""
        out: list[RetrievedChunk] = []
        for position, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                # The index and the chunk table disagree - report it loudly
                # rather than returning a citable id with no text behind it.
                logger.error("fused id %s is missing from the chunk table", chunk_id)
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    page=chunk.page,
                    title=chunk.title,
                    metadata=chunk.metadata,
                    ranks={
                        "sparse": sparse_ranks.get(chunk_id, 0),
                        "dense": dense_ranks.get(chunk_id, 0),
                        "fused": position,
                    },
                    scores={name: scores.get(f"{name}:{chunk_id}", 0.0) for name in ("sparse", "dense")},
                )
            )
        return out

    # -- main entry point ----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: Sequence[Filter] | None = None,
        rerank_results: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve, fuse and (optionally) rerank passages for ``query``."""
        query = (query or "").strip()
        if not query:
            return []
        top_k = self.top_k if top_k is None else max(1, top_k)
        candidate_k = self.candidate_k if candidate_k is None else max(1, candidate_k)
        filters = list(filters or [])
        allowed = self.allowed_rows(filters)

        corpus_size = len(self.store.chunks)
        if allowed is not None and not allowed:
            self.last_debug = RetrievalDebug(0, 0, 0, 0, top_k, self.fusion, None, 0)
            return []

        # Over-fetch from each retriever: fusion can only reorder what it is
        # given, so a passage ranked 40 by BM25 and 2 by dense still needs to be
        # in the BM25 list for the dense evidence to help it.
        pool = min(max(candidate_k, top_k * 4), max(corpus_size, top_k))

        sparse_raw = self.store.bm25.search(query, k=pool, allowed_indices=allowed)
        dense_raw = self.store.search_dense(query, k=pool, allowed_indices=allowed)

        ranked = fuse(
            [
                RankedList("sparse", sparse_raw, self.sparse_weight),
                RankedList("dense", dense_raw, self.dense_weight),
            ],
            strategy="linear" if self.fusion == "linear" else "rrf",
            k=self.rrf_k,
        )

        if not ranked:
            self.last_debug = RetrievalDebug(
                filtered_rows=len(allowed) if allowed is not None else corpus_size,
                dense_candidates=len(dense_raw),
                sparse_candidates=len(sparse_raw),
                fused_candidates=0,
                top_k=top_k,
                fusion=self.fusion,
                reranker=None,
                rerank_applied=0,
            )
            return []

        scores: dict[str, float] = {}
        for name, results in (("sparse", sparse_raw), ("dense", dense_raw)):
            for chunk_id, value in results:
                key = f"{name}:{chunk_id}"
                # A chunk can appear in both lists; keep the better score so the
                # reported number matches the retriever that ranked it highest.
                if key not in scores or value > scores[key]:
                    scores[key] = float(value)

        chunks = self._materialise(ranked, _ranks(sparse_raw), _ranks(dense_raw), scores)

        should_rerank = self.reranker is not None if rerank_results is None else rerank_results
        rerank_applied = 0
        if should_rerank and self.reranker is not None:
            # Rerank the whole fused pool (bounded by candidate_k), then cut.
            head = chunks[: max(candidate_k, top_k)]
            rerank_applied = len(head)
            chunks = rerank(self.reranker, query, head) + chunks[len(head) :]

        self.last_debug = RetrievalDebug(
            filtered_rows=len(allowed) if allowed is not None else corpus_size,
            dense_candidates=len(dense_raw),
            sparse_candidates=len(sparse_raw),
            fused_candidates=len(ranked),
            top_k=top_k,
            fusion=self.fusion,
            reranker=self.reranker.name if (should_rerank and self.reranker) else None,
            rerank_applied=rerank_applied,
        )
        return chunks[:top_k]
