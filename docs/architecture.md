# Architecture

This document explains *why* the pipeline is shaped the way it is. For the
one-paragraph version, see the [README](../README.md); for the numbers, see
[`eval/README.md`](../eval/README.md).

## Overview

```
ingest → index → retrieve → fuse → rerank → generate → cite
```

Each arrow is a module with one job, and each has a dependency-free fallback so
the whole thing runs (and is tested) without FAISS, `torch` or Sentence
Transformers. The fallbacks are deterministic, which is what makes the corpus and
the evaluation reproducible in CI.

## 1. Ingestion (`app/core/ingest.py`, `app/core/chunking.py`)

`ingest_paths` expands files and directories, dispatches by suffix, and returns
`Document` objects with normalised text. PDFs are read page by page with `pypdf`;
a scanned PDF with no text layer is reported as a failure rather than indexed as
an empty document. Front-matter metadata (`year`, `title`, `tags`, …) is extracted
where present.

Chunking (`chunk_text`) splits pages into sentences, then packs sentences into
~`chunk_words`-word windows with `chunk_overlap_words` of overlap. Two details
matter:

- **Sentence boundaries, not raw word counts.** A chunk is not allowed to start
  mid-sentence, so a citation quotes something a human would recognise.
- **A `min_words` guard merges a too-short trailing chunk** into its predecessor,
  which stops the corpus filling up with orphan fragments that pollute the
  candidate list.

Chunks carry `doc_id`, `page`, `title` and metadata through to the citation.

## 2. Indexing (`app/services/store.py`)

`IndexStore` owns three things and keeps them consistent:

| Artefact            | Source of truth           | Persisted |
| ------------------- | ------------------------- | --------- |
| `chunks.jsonl`      | documents + chunking      | yes       |
| `vectors.npy`       | embedder output           | yes       |
| `meta.json`         | dimensions, embedder name | yes       |
| BM25 index          | rebuilt from `chunks.jsonl` | no      |
| FAISS `IndexFlatIP` | rebuilt from `vectors.npy`  | no      |

The two "no" rows are deliberate. BM25 is cheap to rebuild and its parameters
(`k1`, `b`) are config, so persisting it would let a stale `k1` survive a config
change. FAISS is rebuilt from `vectors.npy` because the NumPy array is the
authoritative record — a rebuilt index can never disagree with the vectors it was
built from, whereas a serialised FAISS index can drift. `meta.json` records the
embedding dimension and embedder name; `IndexStore.load` **rejects** an index
whose dimension does not match the configured embedder rather than answering
nonsense from mismatched vectors.

Documents are content-addressed (`blake2b` of title + text), so re-ingesting the
same file is a no-op instead of a duplicate.

### Why `blake2b` and not `hash()`

`hash()` is salted per process, so vectors persisted by one run would be
unreadable (and differently bucketed) in the next. The hashing embedder uses
`hashlib.blake2b` precisely so a persisted index is reproducible across restarts.

## 3. Embeddings (`app/core/embeddings.py`)

Two implementations behind one interface:

- `SentenceTransformerEmbedder` — a pretrained bi-encoder (`all-MiniLM-L6-v2` by
  default). Semantic, ~90 MB download.
- `HashingEmbedder` — signed feature hashing over unigrams and bigrams with
  sublinear (`1 + log tf`) weighting, L2-normalised. Deterministic, NumPy only,
  **and lexical**.

The importance of that last word is easy to miss, so it is spelled out below.

## 4. Vector store (`app/core/vector_store.py`)

`VectorStore` wraps either FAISS `IndexFlatIP` (exact inner-product search — the
vectors are already L2-normalised, so inner product is cosine similarity) or a
NumPy brute-force fallback. Both accept an optional `allowed_indices` list so a
metadata filter and a vector search compose. `.backend` reports `"faiss"` or
`"numpy"` and is surfaced by `/stats`.

## 5. BM25 (`app/core/bm25.py`)

Okapi BM25, pure Python: `idf(q) = ln(1 + (N - df + 0.5)/(df + 0.5))`, with
length normalisation `k1 = 1.5`, `b = 0.75`. No external dependency, so the sparse
retriever always exists.

## 6. Fusion (`app/core/hybrid.py`)

BM25 scores and cosine similarities live on incomparable scales, so fusion
operates on **ranks**:

- **RRF (default):** `score(d) = Σ_r w_r / (k + rank_r(d))`, `k = 60`. Scale-free,
  no tuning, robust to one retriever returning garbage.
- **Linear:** min-max normalise each list to `[0, 1]`, then weight. Keeps absolute
  confidence (a document both retrievers like can win) at the cost of outlier
  sensitivity in a short list.

`minmax` maps a constant list to all-ones rather than all-zeros — a subtlety the
reranker depends on (see below).

### The lexical-proxy dense cap

RRF assumes its inputs contribute **independent** evidence. The hashing embedder
does not: it ranks largely by shared terms, so it is strongly correlated with
BM25. At equal weights two correlated rankings can outvote the better-calibrated
sparse one.

So when `HashingEmbedder` is live, `RAGAssistant._effective_dense_weight` caps the
dense weight at `RAG_LEXICAL_DENSE_WEIGHT` (default `0.5`). Measured on the
labelled set: recall@1 `0.786 → 0.810`, MRR `0.925 → 0.937`. A real bi-encoder is
uncorrelated with BM25 and keeps the full weight. The cap lives in
`RAGAssistant`, not in `reciprocal_rank_fusion`, so the fusion function stays a
pure function of the weights it is handed — and the effective weights are reported
by `/health` and `/stats` rather than being a hidden constant.

## 7. Retriever (`app/core/retriever.py`)

`HybridRetriever.retrieve` is the orchestration point:

1. **Resolve the filter once** into allowed row indices (`allowed_rows`, public so
   the evaluation baselines run over the identical candidate set).
2. **Over-fetch** `candidate_k` from each retriever. Fusion can only reorder what
   it is given: a passage ranked 40th by BM25 and 2nd by dense still has to be in
   the BM25 list for the dense evidence to help it.
3. **Fuse** into one pool.
4. **Rerank the pool, not the final `top_k`.** The candidate head (bounded by
   `candidate_k`) is reranked and then cut. Reranking only the five already-chosen
   passages would let the cross-encoder shuffle the answer but never fix it.
5. **Materialise** full chunk records, attaching `ranks` (`sparse`/`dense`/`fused`)
   and `retriever_scores` for the debug output.

`RetrievalDebug` records per-stage counts and is returned by `debug=true`.

## 8. Reranking (`app/core/reranker.py`)

- `CrossEncoderReranker` — `ms-marco-MiniLM-L-6-v2` scores each `(query, passage)`
  pair jointly. Accurate, far too slow to run over a corpus, so it runs on the
  retrieval head only.
- `LexicalReranker` — IDF-weighted query-term coverage plus a proximity bonus,
  blended with the (min-max normalised) retrieval score.

Two traps the lexical reranker is written around:

- **Do not squash the retrieval signal.** RRF scores are ~`1/60` in magnitude, so
  `tanh`/`sigmoid` maps them all to nearly the same value and the retrieval signal
  vanishes — "rerank" silently becomes "discard the hybrid ranking". The blend
  normalises across the candidate list with `minmax` first.
- **Keep the weight small and measured.** The shipped default (`0.1`) was swept on
  the labelled set and is *neutral*: with a lexical proxy embedder and a lexical
  reranker there is no non-lexical signal to add, and higher weights trade nDCG@5
  and MAP away for a flat rank-1 swap. The reranking stage pays off only with the
  real cross-encoder — see [`eval/README.md`](../eval/README.md).

## 9. Generation and citations (`app/core/generator.py`)

- `extractive` (default): selects sentences from the top passages that share
  content terms with the question and quotes them. It **refuses** to answer when
  no passage shares a content term (`min_overlap_terms=1`), which is the honest
  behaviour for a retrieval demo — no hallucination, no API key.
- `openai`: sends the passages to any OpenAI-compatible chat endpoint.

`build_citations` maps each used passage to a numbered citation with a doc title,
page and snippet, so an answer is traceable to text.

## 10. Evaluation (`app/evaluation/metrics.py`)

Metrics: Recall@K, hit-rate@K, precision@K, nDCG@K, MRR, MAP. The design choice
worth calling out is **anchor labelling**:

> A test label is an exact sentence from a source document, not a chunk id. At
> evaluation time `resolve_anchors` finds whichever chunk *currently* contains that
> sentence and treats it as gold.

Chunk ids are an artefact of chunking parameters; a label that hard-codes one goes
silently stale the moment `chunk_words` changes. Sentence anchors survive
re-chunking. A label whose anchor can no longer be found is reported as
unresolved and skipped — a stale label is an evaluation bug, not a retrieval
failure.

`index_coverage` reports how many labels resolved, so a test set that has drifted
away from the corpus is visible rather than quietly scoring zero.

## 11. Service and API (`app/services/assistant.py`, `app/api/`)

`RAGAssistant` owns the index lifecycle and the end-to-end query path. Two
construction paths are explicit: `load` (the server) and `build` (the scripts).
`load_or_build` falls back to the bundled sample corpus so a fresh clone answers a
question instead of returning 503.

`app/main.py` builds the index in a FastAPI `lifespan` handler rather than at
import time — a module-level load would download a model the moment anything
imported the app, including the tests. Startup failures are caught and logged;
a broken index yields an empty-but-honest service instead of a container that
exits on boot.

The route handlers are thin by design: all decisions (fusion, filtering, ranking)
live in `app.core`, so the API, the CLI scripts and the tests exercise the same
code paths.

## Fallback matrix

| Component  | Preferred                        | Fallback                          | Reported by            |
| ---------- | -------------------------------- | --------------------------------- | ---------------------- |
| Embedding  | `sentence-transformers`          | hashing (blake2b, unigram+bigram) | `/health`, `/stats`    |
| Vector idx | FAISS `IndexFlatIP`              | NumPy brute force                 | `/stats`               |
| Reranking  | `cross-encoder/ms-marco-…`       | lexical coverage + proximity      | `/health`, `/stats`    |
| Generation | OpenAI-compatible endpoint       | extractive quote                  | `/health`              |
| BM25       | always pure Python               | —                                 | `/stats` (k1, b)       |

Every fallback is named in an API response. A number is never silently produced by
a weaker code path.
