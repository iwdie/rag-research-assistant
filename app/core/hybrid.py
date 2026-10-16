"""Hybrid retrieval: fuse BM25 and dense rankings into one candidate list.

Two fusion strategies, both operating on *ranks* rather than raw scores, since
BM25 scores and cosine similarities live on incomparable scales:

``rrf`` (default)
    Reciprocal Rank Fusion: ``score(d) = sum_r w_r / (k + rank_r(d))``. Scale
    free, needs no tuning, and robust when one retriever returns garbage. The
    ``k`` constant (default 60) flattens the head of each list so the second hit
    is not treated as half as good as the first.

``linear``
    Min-max normalise each retriever's scores to [0, 1], then take a weighted
    sum. Keeps absolute confidence information - a document that both retrievers
    score highly can outrank a document that tops only one list - at the cost of
    being sensitive to outliers in a short candidate list.

The union of both candidate lists is returned, so ``candidate_k`` is an upper
bound on the fused pool rather than a per-retriever cap.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class RetrievedChunk:
    """One candidate passage plus the provenance of its score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    page: int | None = None
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    rerank_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "score": round(self.score, 6),
            "ranks": self.ranks,
            "retriever_scores": {k: round(v, 6) for k, v in self.scores.items()},
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RankedList:
    """A retriever's output, tagged so fusion can weight and diagnose it."""

    name: str
    results: Sequence[tuple[str, float]]
    weight: float = 1.0


def reciprocal_rank_fusion(rankings: Iterable[RankedList], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists with RRF. Returns ids sorted by descending score."""
    fused: dict[str, float] = {}
    for ranked in rankings:
        if ranked.weight == 0.0:
            continue
        for rank, (doc_id, _score) in enumerate(ranked.results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + ranked.weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def minmax(scores: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1]; a constant list maps to all-ones, not all-zeros."""
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.ones_like(scores)
    return (scores - low) / (high - low)


def linear_fusion(rankings: Iterable[RankedList]) -> list[tuple[str, float]]:
    """Weighted sum of min-max normalised scores."""
    weighted: dict[str, float] = {}
    for ranked in rankings:
        if ranked.weight == 0.0 or not ranked.results:
            continue
        ids = [doc_id for doc_id, _ in ranked.results]
        scores = minmax(np.asarray([score for _, score in ranked.results], dtype=np.float64))
        for doc_id, score in zip(ids, scores, strict=False):
            weighted[doc_id] = weighted.get(doc_id, 0.0) + ranked.weight * float(score)
    return sorted(weighted.items(), key=lambda item: (-item[1], item[0]))


def fuse(
    rankings: Iterable[RankedList],
    *,
    strategy: Literal["rrf", "linear"] = "rrf",
    k: int = 60,
) -> list[tuple[str, float]]:
    """Dispatch to the configured fusion strategy."""
    rankings = list(rankings)
    if strategy == "linear":
        return linear_fusion(rankings)
    return reciprocal_rank_fusion(rankings, k=k)
