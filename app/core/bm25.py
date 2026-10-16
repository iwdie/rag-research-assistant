"""Okapi BM25 over the same chunks the dense index holds.

Written out rather than pulled from a package for two reasons: the sparse
retriever has to support the same filtered-query contract as the dense store
(score only the allowed rows), and BM25 with an explicit ``k1``/``b`` is short
enough that keeping it in-tree makes the hybrid weighting auditable.

Three details that are easy to get wrong:

* IDF is floored at a small epsilon. With more query terms than documents
  containing them, the classic ``log((N - n + 0.5)/(n + 0.5))`` goes negative
  and the ranker starts *penalising* rare terms - the opposite of the intent.
* Scoring is accumulated only over documents in the postings lists, so a query
  with no matching terms costs nothing instead of a full scan.
* The candidate set is applied as a mask over an array sized to the corpus, so
  filtered and unfiltered queries take the same code path.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# A light stop list. "The effect of X on Y" and "effect X Y" should rank alike;
# leaving function words in inflates document length and dilutes ``b``.
STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be
    because been before being below between both but by can cannot could did do
    does doing down during each few for from further had has have having he her
    here hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other ought our
    ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves""".split()
)


def tokenize(text: str, *, keep_stopwords: bool = False, min_length: int = 2) -> list[str]:
    """Lowercase word tokens, optionally dropping stopwords and 1-char noise."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    tokens = [t for t in tokens if len(t) >= min_length or t.isdigit()]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS]


@dataclass
class BM25Index:
    """An inverted index with BM25 scoring and filtered candidate support."""

    k1: float = 1.5
    b: float = 0.75
    ids: list[str] = field(default_factory=list)
    _postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _avgdl: float = field(default=0.0, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    # -- construction --------------------------------------------------------
    def fit(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        self.ids = list(ids)
        self._postings = defaultdict(dict)
        self._lengths = []
        self._idf = {}

        for row, text in enumerate(texts):
            tokens = tokenize(text)
            self._lengths.append(len(tokens))
            for token in tokens:
                bucket = self._postings[token]
                bucket[row] = bucket.get(row, 0) + 1

        total = sum(self._lengths)
        self._avgdl = total / len(self._lengths) if self._lengths else 0.0
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        n_docs = len(self.ids)
        self._idf = {}
        for token, bucket in self._postings.items():
            df = len(bucket)
            self._idf[token] = max(
                math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)),
                1e-6,
            )

    def extend(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        """Add documents, recomputing IDF (cheap: one pass over the corpus)."""
        merged_ids = self.ids + list(ids)
        start = len(self.ids)
        for offset, text in enumerate(texts):
            row = start + offset
            tokens = tokenize(text)
            self._lengths.append(len(tokens))
            for token in tokens:
                bucket = self._postings[token]
                bucket[row] = bucket.get(row, 0) + 1
        self.ids = merged_ids
        total = sum(self._lengths)
        self._avgdl = total / len(self._lengths) if self._lengths else 0.0
        self._compute_idf()
        return self

    # -- querying ------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.ids)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @property
    def vocabulary(self) -> int:
        return len(self._postings)

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        allowed_indices: Iterable[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, score)`` pairs, best first.

        ``allowed_indices`` (positions into ``self.ids``) restricts scoring to a
        filtered candidate set. Pass ``None`` for an unrestricted query - note
        that ``[]`` means "nothing is allowed" and correctly returns nothing.
        """
        if not self.ids or k <= 0:
            return []
        allowed = None if allowed_indices is None else set(allowed_indices)
        if allowed is not None and not allowed:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float64)
        touched = False
        for term in set(query_terms):
            bucket = self._postings.get(term)
            if not bucket:
                continue
            touched = True
            idf = self._idf.get(term, 0.0)
            for row, tf in bucket.items():
                if allowed is not None and row not in allowed:
                    continue
                length = self._lengths[row] or 1
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
                scores[row] += idf * (tf * (self.k1 + 1)) / denominator

        if not touched:
            return []

        limit = min(k, self.n_docs)
        # argpartition is O(n) instead of O(n log n); we only need the top slice
        # ordered, not the whole corpus.
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.ids[row], float(scores[row])) for row in top if scores[row] > 0.0]

    # -- persistence ---------------------------------------------------------
    def to_state(self) -> dict:
        """Serialise only what cannot be rebuilt from the chunk texts."""
        return {"k1": self.k1, "b": self.b, "ids": self.ids}

    @classmethod
    def from_state(cls, state: dict, texts: Sequence[str]) -> BM25Index:
        index = cls(k1=float(state.get("k1", 1.5)), b=float(state.get("b", 0.75)))
        return index.fit(list(state["ids"]), texts)
