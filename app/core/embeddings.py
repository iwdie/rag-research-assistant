"""Dense text embeddings with a dependency-free fallback.

Two backends implement the same tiny interface:

``SentenceTransformerEmbedder``
    The real thing - a pretrained bi-encoder. Downloads ~90 MB on first use.

``HashingEmbedder``
    A signed feature-hashing bag of words (unigrams + bigrams) with sublinear
    term weighting, L2-normalised. It is *not* semantic, but it is deterministic,
    needs nothing but NumPy, and is good enough to exercise ingestion, hybrid
    fusion, filtering, reranking and the whole API in tests and CI.

Having the fallback is what makes the retrieval plumbing testable: a broken
fusion rule or an off-by-one in the metric code shows up in CI instead of being
hidden behind a multi-hundred-megabyte download.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Deliberately small: the hashing embedder exists to be fast and predictable.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with we our their they he she his her not but can may""".split()
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit-norm float32 vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of L2-normalised rows."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe for zero rows.

    Normalising means inner product == cosine similarity, so a FAISS
    ``IndexFlatIP`` is an exact cosine index and no metric conversion is needed
    between the FAISS path and the NumPy fallback.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """Map a token to a bucket index and a sign.

    Two independent halves of one BLAKE2b digest. Python's built-in ``hash()``
    is salted per process, which would make persisted vectors unreadable after a
    restart - the whole point of hashing here is that it is reproducible.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big")
    index = raw % dim
    sign = 1.0 if (raw >> 63) & 1 else -1.0
    return index, sign


def _tokens(text: str) -> list[str]:
    words = [token.lower() for token in _TOKEN_RE.findall(text)]
    kept = [w for w in words if w not in _STOPWORDS or len(words) <= 4]
    return kept or words


class HashingEmbedder:
    """Signed feature hashing over unigrams and bigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-blake2b-{dim}"

    def _features(self, text: str) -> dict[str, float]:
        words = _tokens(text)
        counts: dict[str, float] = {}
        for word in words:
            counts[word] = counts.get(word, 0.0) + 1.0
        for first, second in pairwise(words):
            bigram = f"{first}_{second}"
            # Bigrams are worth less than their parts: they disambiguate, they
            # should not dominate the similarity.
            counts[bigram] = counts.get(bigram, 0.0) + 0.5
        return counts

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token, count in self._features(text or "").items():
                index, sign = _bucket(token, self.dim)
                # sqrt-ish weighting: the 10th mention of a term matters, but
                # not ten times as much as the first.
                matrix[row, index] += sign * (1.0 + np.log(count))
        return _l2_normalise(matrix)


class SentenceTransformerEmbedder:
    """Thin wrapper over ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # encode(normalize_embeddings=True) already normalises; re-normalising is
        # free insurance against a model that ignores the flag.
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder(
    *,
    backend: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hashing_dim: int = 512,
    batch_size: int = 32,
) -> Embedder:
    """Build an embedder, degrading to hashing when Sentence Transformers is absent."""
    if backend == "fallback":
        return HashingEmbedder(dim=hashing_dim)
    try:
        return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
    except Exception as exc:
        if backend == "native":
            raise
        logger.warning(
            "sentence-transformers unavailable (%s: %s); using the hashing fallback. "
            "Retrieval quality will be lexical, not semantic.",
            type(exc).__name__,
            exc,
        )
        return HashingEmbedder(dim=hashing_dim)
