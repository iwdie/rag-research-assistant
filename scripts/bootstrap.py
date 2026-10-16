"""Bootstrap a working demo in one command.

``python scripts/bootstrap.py``

Generates the sample corpus, builds the index and prints a smoke query with its
citations - the fastest way to check that the whole pipeline runs on a machine
before wiring anything to it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings
from app.services.assistant import RAGAssistant


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from scripts.make_sample_data import main as make_sample

    if make_sample([]) != 0:
        return 1

    settings = Settings()
    assistant = RAGAssistant.build(settings, paths=[REPO_ROOT / "data" / "sample"])
    assistant.save()

    question = "How does hybrid retrieval combine sparse and dense scores?"
    result = assistant.query(question, debug=True)
    print("\n" + "=" * 78)
    print(f"Q: {question}")
    print("-" * 78)
    print(result.answer.answer)
    print("-" * 78)
    for citation in result.answer.citations:
        location = f", p.{citation.page}" if citation.page else ""
        print(f"[{citation.marker}] {citation.title}{location}  ({citation.chunk_id}, score {citation.score:.3f})")
    if result.debug:
        print("-" * 78)
        print(f"debug: {result.debug}")
    print("=" * 78)
    print("\nNext: uvicorn app.main:app --reload  (then open http://127.0.0.1:8000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
