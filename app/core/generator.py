"""Answer synthesis with inline source citations.

``ExtractiveGenerator`` (default)
    No model, no API key. It splits the selected passages into sentences, scores
    each against the query, and stitches the best few into a short answer with
    ``[n]`` markers pointing at the citation list. Every sentence in the output
    is verbatim from a retrieved passage, which means the answers cannot
    hallucinate - the failure mode is "unhelpful", never "invented a citation".

    It refuses to answer when no selected passage shares a single content term
    with the question, returning :data:`_NO_ANSWER` instead of quoting something
    unrelated. That guard is deliberately conservative and lexical: it is a
    *floor*, not a calibrated relevance model, so a query phrased entirely in
    vocabulary absent from the corpus gets an honest refusal. With the fallback
    hashing embedder that is the only signal available; with a real bi-encoder
    the dense ranking carries most of the weight and this guard almost never
    fires. Passing ``min_overlap_terms=0`` disables it.

``OpenAIGenerator``
    Calls an OpenAI-compatible chat completion with the passages numbered and an
    instruction to cite. Any failure (no key, timeout, bad status) falls back to
    the extractive answer and says so in ``generator`` on the response, so the
    API never returns a 500 just because a model provider is down.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.bm25 import tokenize
from app.core.chunking import split_sentences
from app.core.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)

_NO_ANSWER = (
    "No passage in the indexed corpus matched this question. Try rephrasing, "
    "widening the metadata filter, or indexing more documents."
)

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered "
    "context passages. Cite every claim with bracketed passage numbers, like [1] "
    "or [2][3]. If the context does not contain the answer, say so plainly. "
    "Be concise: at most four sentences."
)


@dataclass
class Citation:
    """A pointer from an answer back to a passage."""

    marker: int
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    snippet: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "snippet": self.snippet,
            "score": round(self.score, 6),
        }


@dataclass
class Answer:
    """A generated answer and everything needed to audit it."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    generator: str = "extractive"
    used_chunks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "generator": self.generator,
            "used_chunks": self.used_chunks,
        }


def _snippet(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def build_citations(chunks: Sequence[RetrievedChunk], *, limit: int = 5) -> list[Citation]:
    """Number the selected passages 1..n in ranked order."""
    citations = []
    for marker, chunk in enumerate(chunks[:limit], start=1):
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title or chunk.doc_id,
                page=chunk.page,
                snippet=_snippet(chunk.text),
                score=chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
            )
        )
    return citations


class ExtractiveGenerator:
    """Sentence-level extractive answering with citations."""

    name = "extractive"
    max_sentences: int = 3
    min_length: int = 40
    min_overlap_terms: int = 1

    def generate(self, question: str, chunks: Sequence[RetrievedChunk], *, limit: int = 5) -> Answer:
        selected = list(chunks[:limit])
        if not selected:
            return Answer(answer=_NO_ANSWER, citations=[], generator=self.name)

        terms = {t for t in tokenize(question) if len(t) > 2}
        if terms and self.min_overlap_terms > 0:
            shared = {
                term
                for chunk in selected
                for term in terms & set(tokenize(chunk.text))
            }
            if len(shared) < self.min_overlap_terms:
                # Nothing retrieved shares vocabulary with the question. Say so
                # rather than quoting a passage that merely ranked highest.
                return Answer(answer=_NO_ANSWER, citations=[], generator=self.name)

        citations = build_citations(selected, limit=limit)
        candidates: list[tuple[float, int, str]] = []
        for index, chunk in enumerate(selected):
            # A passage ranked first is evidence in itself, so rank contributes
            # alongside lexical overlap with the question.
            prior = 1.0 / (1.0 + index)
            for sentence in split_sentences(chunk.text):
                if len(sentence) < self.min_length:
                    continue
                tokens = set(tokenize(sentence))
                overlap = len(terms & tokens) / (len(terms) or 1)
                candidates.append((0.7 * overlap + 0.3 * prior, index + 1, sentence))

        if not candidates:
            # Fall back to the top passage's opening line: the corpus matched,
            # we just could not find a sentence-shaped answer in it.
            lead = _snippet(selected[0].text, 300)
            return Answer(
                answer=f"[1] {lead}",
                citations=citations,
                generator=self.name,
                used_chunks=[selected[0].chunk_id],
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[tuple[int, str]] = []
        seen_sentences: set[str] = set()
        for _score, marker, sentence in candidates:
            key = sentence.casefold()
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            chosen.append((marker, sentence))
            if len(chosen) >= self.max_sentences:
                break

        chosen.sort(key=lambda item: item[0])
        body = " ".join(f"[{marker}] {sentence.strip()}" for marker, sentence in chosen)
        used = [selected[marker - 1].chunk_id for marker, _ in chosen]
        return Answer(answer=body, citations=citations, generator=self.name, used_chunks=used)


class OpenAIGenerator:
    """Chat-completion answering, with the extractive generator as a safety net."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fallback = ExtractiveGenerator()

    def _prompt(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        passages = "\n\n".join(
            f"[{marker}] ({chunk.title or chunk.doc_id}"
            f"{f', p.{chunk.page}' if chunk.page else ''})\n{chunk.text}"
            for marker, chunk in enumerate(chunks, start=1)
        )
        return f"Context passages:\n\n{passages}\n\nQuestion: {question}"

    def generate(self, question: str, chunks: Sequence[RetrievedChunk], *, limit: int = 5) -> Answer:
        import httpx

        selected = list(chunks[:limit])
        citations = build_citations(selected, limit=limit)
        if not selected:
            return Answer(answer=_NO_ANSWER, citations=[], generator=self.name)

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._prompt(question, selected)},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("OpenAI generation failed (%s: %s); using extractive answer", type(exc).__name__, exc)
            fallback = self._fallback.generate(question, selected, limit=limit)
            fallback.generator = f"{self.name}->extractive"
            return fallback

        used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= len(selected)})
        return Answer(
            answer=text,
            citations=citations,
            generator=self.name,
            used_chunks=[selected[m - 1].chunk_id for m in used],
        )


def get_generator(
    *,
    kind: str = "extractive",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    timeout: float = 30.0,
):
    """Pick a generator, falling back to extractive when the key is missing."""
    if kind == "openai":
        if not api_key:
            logger.warning("generator=openai but no API key configured; using extractive answers")
            return ExtractiveGenerator()
        return OpenAIGenerator(api_key=api_key, model=model, base_url=base_url, timeout=timeout)
    return ExtractiveGenerator()
