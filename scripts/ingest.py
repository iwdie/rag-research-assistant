"""Ingest a corpus: ``python scripts/ingest.py data/sample --sample``.

Adds to the existing index when there is one (documents are content-addressed,
so re-running is idempotent) and saves the result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings
from app.core.ingest import expand_paths
from app.services.assistant import RAGAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="PDF/txt/md files or directories")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="ingest the bundled synthetic corpus (data/sample) instead of paths",
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="defaults to RAG_INDEX_DIR")
    parser.add_argument("--chunk-words", type=int, default=None)
    parser.add_argument("--overlap-words", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="ignore any existing index and start empty")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    overrides: dict = {}
    if args.index_dir:
        overrides["index_dir"] = args.index_dir
    if args.chunk_words:
        overrides["chunk_words"] = args.chunk_words
    if args.overlap_words is not None:
        overrides["chunk_overlap_words"] = args.overlap_words
    settings = Settings(**overrides)

    paths = [REPO_ROOT / "data" / "sample"] if args.sample else args.paths
    if not paths:
        print("nothing to ingest: pass paths or --sample", file=sys.stderr)
        return 2
    if not expand_paths(paths):
        print(f"no supported documents found in {paths}", file=sys.stderr)
        return 1

    if args.rebuild:
        assistant = RAGAssistant.build(settings)
    else:
        try:
            assistant = RAGAssistant.load(settings)
        except (FileNotFoundError, ValueError):
            assistant = RAGAssistant.build(settings)

    added = assistant.index_paths(paths)
    print(f"indexed {len(expand_paths(paths))} files -> {added} new chunks")
    assistant.save()
    stats = assistant.store.stats()
    print(
        f"index now holds {stats['n_documents']} documents / {stats['n_chunks']} chunks "
        f"({stats['total_words']} words), embedder={stats['embedder']}, vectors={stats['vector_backend']}"
    )
    print(f"saved to {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
