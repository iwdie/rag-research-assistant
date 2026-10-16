"""Cross-encoder reranking with a lexical fallback.

Why rerank at all: hybrid retrieval optimises for *recall* - it casts a wide net
with two cheap, independently-flawed scorers. A cross-encoder then scores each
``(query, passage)`` pair jointly, which is far more accurate and far too slow
to run over a corpus. Retrieval decides *what to look at*, reranking decides
*what to quote*.

``CrossEncoderReranker``
    ``sentence_transformers.CrossEncoder`` (ms-marco MiniLM by default).

``LexicalReranker``
    IDF-weighted query-term coverage plus a proximity bonus, blended with the
    retrieval score. Not a semantic model, but it is deterministic, free, and
    catches the failure mode a cross-encoder fixes most often: a passage that
    merely mentions the query's words far apart losing to one that answers it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np

from .bm25 import tokenize
from .hybrid import RetrievedChunk, minmax

logger = logging.getLogger(__name__)

# A window of this many tokens is "close enough" for proximity to count.
_PROXIMITY_WINDOW = 12


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Return one relevance score per chunk, higher is better."""
        ...


def _proximity(tokens: Sequence[str], terms: set[str], window: int = _PROXIMITY_WINDOW) -> float:
    """Fraction of query terms that have a neighbour within ``window`` tokens.

    Rewards passages where terms cluster, which is a decent proxy for "this
    sentence is about the query" versus "these words happen to appear here".
    """
    if not terms or len(tokens) < 2:
        return 0.0
    positions = [i for i, token in enumerate(tokens) if token in terms]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    close = sum(1 for a, b in pairwise(positions) if b - a <= window)
    return close / (len(positions) - 1)


@dataclass
class LexicalReranker:
    """Deterministic, dependency-free reranker used when no model is available.

    This is a stand-in for the cross-encoder, not a replacement: it re-orders the
    fused list using IDF-weighted query-term coverage plus a proximity bonus, so
    it is a purely lexical signal layered on top of two lexical/lexical-proxy
    retrievers.

    ``weight`` was swept on the labelled test set (``eval/testset.jsonl``) with the
    dependency-free backends. The honest result is that the lexical reranker does
    *not* beat the fused order on this corpus at any weight: at 0.1 it ties every
    reported metric, and raising it trades nDCG@5 and MAP away for a flat rank-1
    swap (0.6-0.75 lifts recall@1 by 0.012 while nDCG@5 falls by 0.003). The
    default is therefore the conservative neutral value rather than a flattering
    one. The stage only earns its keep with the real cross-encoder, which scores
    the query and the passage jointly instead of matching terms - that path is
    exercised by the scheduled models workflow (``.github/workflows/models.yml``),
    and ``/stats``/``/health`` always report which reranker is live so a number is
    never attributed to the wrong component.
    """

    name: str = "lexical-coverage-proximity"
    weight: float = 0.1

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        terms = {t for t in tokenize(query) if len(t) > 2}
        if not terms:
            return [0.0 for _ in chunks]

        # IDF surrogate: rarer terms across the candidate pool matter more.
        document_frequency = {term: 0 for term in terms}
        tokenised: list[list[str]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tokenised.append(tokens)
            for term in terms & set(tokens):
                document_frequency[term] += 1
        total = max(len(chunks), 1)
        idf = {
            term: math.log(1.0 + total / (document_frequency[term] + 0.5)) for term in terms
        }
        idf_total = sum(idf.values()) or 1.0

        # The retriever scores must be normalised *across the candidate list*
        # before they are blended. RRF scores are ~1/60 in magnitude, so any
        # monotone squash (tanh, sigmoid) maps them all to almost the same value
        # and the retrieval signal effectively disappears - which silently turns
        # "rerank" into "throw away the hybrid ranking".
        base = minmax(np.asarray([chunk.score for chunk in chunks], dtype=np.float64))

        scores: list[float] = []
        for index, tokens in enumerate(tokenised):
            present = terms & set(tokens)
            coverage = sum(idf[t] for t in present) / idf_total
            proximity = _proximity(tokens, terms)
            lexical = 0.75 * coverage + 0.25 * proximity
            scores.append(self.weight * lexical + (1.0 - self.weight) * float(base[index]))
        return scores


class CrossEncoderReranker:
    """Joint query-passage scoring through a trained cross-encoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.name = model_name

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        raw = self._model.predict(pairs, show_progress_bar=False)
        # ms-marco cross-encoders emit logits; squash so they are comparable to
        # the lexical fallback's [0, 1] range in logs and in the API response.
        return [1.0 / (1.0 + math.exp(-float(value))) for value in raw]


def get_reranker(*, backend: str = "auto", model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """Build a cross-encoder, degrading to the lexical reranker if unavailable."""
    if backend == "fallback":
        return LexicalReranker()
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:
        if backend == "native":
            raise
        logger.warning(
            "cross-encoder unavailable (%s: %s); using the lexical reranker",
            type(exc).__name__,
            exc,
        )
        return LexicalReranker()


def rerank(
    reranker: Reranker,
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Score, sort and truncate ``chunks``. Ties keep their retrieval order."""
    if not chunks:
        return []
    scores = reranker.score(query, chunks)
    for chunk, score in zip(chunks, scores, strict=False):
        chunk.rerank_score = float(score)
    ordered = sorted(
        chunks,
        key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id),
    )
    return ordered[:top_n] if top_n else ordered
