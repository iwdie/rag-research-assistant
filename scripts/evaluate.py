"""Evaluate retrieval: ``python scripts/evaluate.py --compare``.

Prints Recall@K / MRR / nDCG@K for the hybrid pipeline and, with ``--compare``,
for each retriever alone. Anchors in the test set are resolved against the
*current* index, so the numbers always describe the chunking that is loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings
from app.core.filters import parse_filters
from app.evaluation.metrics import (
    DEFAULT_K_VALUES,
    evaluate_ranking,
    index_coverage,
    load_testset,
    resolve_anchors,
)
from app.services.assistant import RAGAssistant

logger = logging.getLogger("evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=REPO_ROOT / "eval" / "testset.jsonl")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="*", default=list(DEFAULT_K_VALUES), help="K cut-offs to report")
    parser.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder / lexical reranker")
    parser.add_argument("--fusion", choices=["rrf", "linear"], default=None, help="override the fusion strategy")
    parser.add_argument("--compare", action="store_true", help="also score sparse-only and dense-only rankings")
    parser.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--show-failures", type=int, default=0, help="print this many misses")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _table(report: Any, keys: list[str]) -> None:
    print(report.table())
    for key in keys:
        if key not in report.metrics:
            print(f"  (missing metric {key})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    overrides: dict = {}
    if args.index_dir:
        overrides["index_dir"] = args.index_dir
    if args.fusion:
        overrides["fusion"] = args.fusion
    settings = Settings(**overrides)

    try:
        assistant = RAGAssistant.load(settings)
    except (FileNotFoundError, ValueError) as exc:
        print(f"could not load an index from {settings.index_dir}: {exc}", file=sys.stderr)
        print("run: python scripts/make_sample_data.py && python scripts/ingest.py --sample", file=sys.stderr)
        return 1

    if assistant.store.n_chunks == 0:
        print("the index is empty; ingest a corpus first", file=sys.stderr)
        return 1

    records = resolve_anchors(load_testset(args.testset), assistant.store.chunks)
    coverage = index_coverage(assistant.store.chunks, records)
    scored = [record for record in records if record.get("relevant")]
    unresolved = [record["query"] for record in records if not record.get("relevant")]
    if not scored:
        print("no test record resolved to a gold chunk; labels are stale", file=sys.stderr)
        return 1

    k_values = sorted({k for k in args.k if k > 0})
    max_k = max(k_values)
    queries = [record["query"] for record in scored]
    relevant_sets = [set(record["relevant"]) for record in scored]

    def filters_for(record: dict) -> list:
        return parse_filters(record["filters"]) if record.get("filters") else []

    def hybrid(record: dict) -> list[str]:
        return [
            chunk.chunk_id
            for chunk in assistant.retriever.retrieve(
                record["query"],
                top_k=max_k,
                filters=filters_for(record),
                rerank_results=not args.no_rerank,
            )
        ]

    def sparse(record: dict) -> list[str]:
        allowed = assistant.retriever.allowed_rows(filters_for(record))
        return [
            chunk_id
            for chunk_id, _ in assistant.store.bm25.search(record["query"], k=max_k, allowed_indices=allowed)
        ]

    def dense(record: dict) -> list[str]:
        allowed = assistant.retriever.allowed_rows(filters_for(record))
        return [
            chunk_id
            for chunk_id, _ in assistant.store.search_dense(record["query"], k=max_k, allowed_indices=allowed)
        ]

    hybrid_report = evaluate_ranking(
        "hybrid" + ("" if args.no_rerank else "+rerank"),
        queries,
        relevant_sets,
        [hybrid(record) for record in scored],
        k_values=k_values,
    )

    print(f"\nindex      : {settings.index_dir}  ({coverage['n_chunks']} chunks)")
    print(f"embedder   : {assistant.store.embedder.name} via {assistant.store.vectors.backend}")
    print(f"reranker   : {(assistant.reranker.name if assistant.reranker else 'disabled') if not args.no_rerank else 'disabled'}")
    print(f"fusion     : {settings.fusion}")
    print(f"test set   : {args.testset}  ({coverage['n_queries']} queries, {coverage['n_labels']} gold chunks)")
    if unresolved:
        print(f"!! {len(unresolved)} queries could not be resolved to a gold chunk and were skipped")
    print()

    # Baselines first, the pipeline under test last, so `reports[:-1]` is
    # exactly "everything we are claiming to beat".
    reports = []
    if args.compare:
        reports.append(
            evaluate_ranking("sparse_bm25", queries, relevant_sets, [sparse(r) for r in scored], k_values=k_values)
        )
        reports.append(
            evaluate_ranking("dense_only", queries, relevant_sets, [dense(r) for r in scored], k_values=k_values)
        )
        if not args.no_rerank:
            # Isolates the reranker: same fusion, same candidates, no reranking.
            reports.append(
                evaluate_ranking(
                    "hybrid_no_rerank",
                    queries,
                    relevant_sets,
                    [
                        [
                            chunk.chunk_id
                            for chunk in assistant.retriever.retrieve(
                                record["query"],
                                top_k=max_k,
                                filters=filters_for(record),
                                rerank_results=False,
                            )
                        ]
                        for record in scored
                    ],
                    k_values=k_values,
                )
            )
    reports.append(hybrid_report)

    for report in reports:
        print(report.table())
        print()

    # The single most useful block in the output: per metric, did hybrid +
    # rerank beat the best simpler configuration, or is the extra machinery
    # dead weight? A pipeline that only wins on recall@5 has bought coverage by
    # making the ranking worse, which this makes visible.
    if args.compare:
        summarised = [m for m in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5") if m in hybrid_report.metrics]
        print("vs best baseline, per metric:")
        for metric in summarised:
            baselines = {report.name: report.metrics.get(metric, 0.0) for report in reports[:-1]}
            best_name = max(baselines, key=lambda name: baselines[name])
            delta = hybrid_report.metrics[metric] - baselines[best_name]
            verdict = "better" if delta > 1e-9 else ("equal" if abs(delta) < 1e-9 else "worse")
            print(
                f"  {metric:<10} {hybrid_report.metrics[metric]:.4f}  "
                f"best={best_name} {baselines[best_name]:.4f}  -> {verdict} ({delta:+.4f})"
            )

    if args.show_failures:
        misses = [q for q in hybrid_report.per_query if q.missed][: args.show_failures]
        if misses:
            print(f"\nfirst {len(misses)} misses:")
        for miss in misses:
            print(f"  [{miss.metrics.get('mrr', 0):.2f}] {miss.query}")
            print(f"        gold={miss.relevant} got={miss.retrieved[:3]}")

    if args.json:
        payload = {
            "coverage": coverage,
            "reports": {report.name: report.as_dict() for report in reports},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
