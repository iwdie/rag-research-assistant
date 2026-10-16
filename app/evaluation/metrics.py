"""Retrieval evaluation: Recall@K, MRR, nDCG@K, MAP@K and hit rate.

The metrics are computed on labelled anchors rather than hand-picked chunk ids,
which keeps the evaluation set valid when chunking changes. See
``eval/README.md`` for the labelling convention.

Definitions used here, since "Recall@K" is not one metric:

``recall@k``
    ``|relevant ∩ top_k| / |relevant|``, averaged over queries. With one gold
    passage per query this is the same as hit rate; with several it is not, and
    the distinction matters for a multi-hop question.
``mrr``
    ``1 / rank`` of the first relevant hit, averaged over queries. Rewards
    putting *something* correct at the top, which is what a reranker is for.
``ndcg@k``
    Graded position-aware quality with binary gains and log2 discount,
    normalised by the ideal ranking.
``map@k``
    Mean of precision@i at each relevant hit, over queries.

Every metric is computed against the *fused* ranking, and Recall@K is reported
for both the hybrid pipeline and each retriever alone so the contribution of
fusion and of reranking can be read off the same table.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids found in the top ``k``."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant id is in the top ``k``."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant hit (0 if none)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at ``k``, normalised by the number of relevant ids."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k with a log2 position discount."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class QueryResult:
    """Per-query metrics plus the ranking that produced them."""

    query: str
    retrieved: list[str]
    relevant: list[str]
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def missed(self) -> bool:
        return self.metrics.get("recall@5", 0.0) == 0.0


@dataclass
class EvaluationReport:
    """Aggregate metrics for one configuration."""

    name: str
    n_queries: int
    metrics: dict[str, float]
    per_query: list[QueryResult] = field(default_factory=list)

    def table(self) -> str:
        """A fixed-width table suitable for a README or CI log."""
        keys = sorted(self.metrics)
        width = max(len(k) for k in keys) if keys else 8
        lines = [f"{self.name}  ({self.n_queries} queries)", "-" * (width + 12)]
        for key in keys:
            lines.append(f"{key:<{width}}  {self.metrics[key]:.4f}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "failures": [
                {"query": q.query, "relevant": q.relevant, "retrieved": q.retrieved[:5]}
                for q in self.per_query
                if q.missed
            ],
        }


def evaluate_ranking(
    name: str,
    queries: Sequence[str],
    relevant_sets: Sequence[set[str]],
    retrieved_lists: Sequence[Sequence[str]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> EvaluationReport:
    """Aggregate standard IR metrics over precomputed rankings."""
    if not (len(queries) == len(relevant_sets) == len(retrieved_lists)):
        raise ValueError("queries, relevant_sets and retrieved_lists must be the same length")

    per_query: list[QueryResult] = []
    totals: dict[str, float] = {}
    for query, relevant, retrieved in zip(queries, relevant_sets, retrieved_lists, strict=False):
        metrics: dict[str, float] = {}
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            metrics[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
            metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        max_k = max(k_values)
        metrics["mrr"] = reciprocal_rank(retrieved, relevant)
        metrics[f"map@{max_k}"] = average_precision(retrieved, relevant, max_k)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        per_query.append(
            QueryResult(query=query, retrieved=list(retrieved), relevant=sorted(relevant), metrics=metrics)
        )

    n = len(queries) or 1
    return EvaluationReport(
        name=name,
        n_queries=len(queries),
        metrics={key: value / n for key, value in totals.items()},
        per_query=per_query,
    )


def evaluate_retriever(
    retriever: Any,
    testset: Sequence[dict[str, Any]],
    *,
    name: str = "hybrid+rerank",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    resolve: Any | None = None,
) -> EvaluationReport:
    """Run ``retriever`` over a test set and aggregate the metrics.

    ``resolve(query, record)`` may override the ranking, which is how the
    side-by-side sparse-only / dense-only comparison is produced without
    duplicating the whole evaluation harness.
    """
    queries: list[str] = []
    relevant_sets: list[set[str]] = []
    retrieved_lists: list[list[str]] = []
    skipped = 0

    for record in testset:
        query = record["query"]
        relevant = set(record.get("relevant") or [])
        if not relevant:
            skipped += 1
            continue
        if resolve is not None:
            retrieved = [c.chunk_id for c in resolve(record)]
        else:
            retrieved = [c.chunk_id for c in retriever.retrieve(query, top_k=max(k_values))]
        queries.append(query)
        relevant_sets.append(relevant)
        retrieved_lists.append(retrieved)

    if skipped:
        logger.warning("skipped %d test records without a resolved gold chunk", skipped)
    return evaluate_ranking(name, queries, relevant_sets, retrieved_lists, k_values=k_values)


def load_testset(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL test set. ``relevant`` may be ids or anchor sentences."""
    records: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else payload.get("queries", [])
    else:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                records.append(json.loads(line))
    return records


def resolve_anchors(
    records: Iterable[dict[str, Any]],
    chunks: Sequence[Any],
) -> list[dict[str, Any]]:
    """Turn sentence anchors into gold chunk ids against the current chunking.

    An ``anchor`` is an exact sentence from the source document. The gold chunk
    is whichever chunk contains it. Anchors that no longer appear (document
    changed, sentence split differently) are dropped by the caller's skip path
    rather than scored as a miss - a stale label is an evaluation bug, not a
    retrieval failure.
    """

    def blob(text: str) -> str:
        return " ".join(text.split()).casefold()

    resolved: list[dict[str, Any]] = []
    for record in records:
        if record.get("relevant"):
            resolved.append(record)
            continue
        anchors = record.get("anchors") or ([record["anchor"]] if record.get("anchor") else [])
        gold: list[str] = []
        for anchor in anchors:
            needle = blob(anchor)
            for chunk in chunks:
                if needle in blob(chunk.text):
                    gold.append(chunk.chunk_id)
                    break
        record = dict(record)
        record["relevant"] = sorted(set(gold))
        resolved.append(record)
    return resolved


def index_coverage(
    chunks: Sequence[Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Sanity stats about the test set: label count and gold-chunk spans."""
    labels = sum(len(record.get("relevant") or []) for record in records)
    unresolved = sum(1 for record in records if not record.get("relevant"))
    spans = []
    for record in records:
        if record.get("relevant"):
            spans.append(len(record["relevant"]))
    return {
        "n_chunks": len(chunks),
        "n_queries": len(records),
        "n_labels": labels,
        "unresolved_queries": unresolved,
        "avg_gold_per_query": (sum(spans) / len(spans)) if spans else 0.0,
    }
