"""Dense vector store: FAISS when available, NumPy brute force otherwise.

The store keeps the raw ``(n, dim)`` float32 matrix as the source of truth and
builds the FAISS index over it. That is a deliberate trade: writing a
``faiss.index`` file alongside the matrix invites the two to drift (add a
document, forget to re-save, silently search a stale index). Rebuilding an
``IndexFlatIP`` from the matrix on load costs milliseconds for a corpus of this
size and removes a whole class of bug.

Because all vectors are L2-normalised on the way in, inner product *is* cosine
similarity and the two backends return identical scores for identical inputs.
``tests/test_vector_store.py`` asserts that parity rather than trusting it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Cosine-similarity search over chunk embeddings."""

    def __init__(self, dim: int, *, use_faiss: bool = True) -> None:
        self.dim = int(dim)
        self.ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._index = None
        self.backend = "numpy"
        if use_faiss:
            self._try_init_faiss()

    def _try_init_faiss(self) -> None:
        try:
            import faiss

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.dim)
            self.backend = "faiss"
        except Exception as exc:
            logger.warning("FAISS unavailable (%s: %s); using NumPy brute force", type(exc).__name__, exc)
            self._index = None
            self.backend = "numpy"

    # -- mutation ------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (*, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must be the same length")
        if self._index is not None and vectors.shape[0]:
            self._index.add(vectors)
        self._matrix = np.vstack([self._matrix, vectors]) if vectors.shape[0] else self._matrix
        self.ids.extend(ids)

    def clear(self) -> None:
        self.ids = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        if self._index is not None:
            self._index = self._faiss.IndexFlatIP(self.dim)

    def rebuild(self) -> None:
        """Recompute the FAISS index from the authoritative matrix."""
        if self._index is None:
            return
        self._index = self._faiss.IndexFlatIP(self.dim)
        if self._matrix.shape[0]:
            self._index.add(self._matrix)

    # -- query ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.ids)

    def _normalise(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query has dim {query.shape[0]}, store expects {self.dim}")
        norm = float(np.linalg.norm(query))
        return query / norm if norm else query

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        allowed_indices: Iterable[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, score)`` pairs by descending cosine similarity."""
        if self.size == 0 or k <= 0:
            return []
        query = self._normalise(query)

        if allowed_indices is None:
            if self._index is not None:
                scores, rows = self._index.search(query.reshape(1, -1), min(k, self.size))
                return [
                    (self.ids[int(row)], float(score))
                    for score, row in zip(scores[0], rows[0], strict=False)
                    if row >= 0
                ]
            subset = self._matrix
            rows_all = None
        else:
            rows_all = np.asarray(sorted(set(allowed_indices)), dtype=np.int64)
            if rows_all.size == 0:
                return []
            subset = self._matrix[rows_all]

        scores = subset @ query
        limit = min(k, scores.shape[0])
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            (self.ids[int(rows_all[row])] if rows_all is not None else self.ids[int(row)], float(scores[row]))
            for row in top
        ]

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "vector_ids.json").write_text(
            json.dumps({"dim": self.dim, "ids": self.ids}, indent=2), encoding="utf-8"
        )

    def load(self, directory: Path) -> VectorStore:
        directory = Path(directory)
        payload = json.loads((directory / "vector_ids.json").read_text(encoding="utf-8"))
        matrix = np.load(directory / "vectors.npy").astype(np.float32)
        dim = int(payload["dim"])
        if matrix.size and matrix.shape[1] != self.dim:
            raise ValueError(
                f"persisted vectors are dim {matrix.shape[1]} but the embedder produces "
                f"{self.dim}. Re-index, or point RAG_INDEX_DIR at the matching index."
            )
        self.dim = dim
        self.ids = list(payload["ids"])
        self._matrix = matrix.reshape(len(self.ids), dim)
        if getattr(self, "_faiss", None) is not None:
            self.rebuild()
        return self
