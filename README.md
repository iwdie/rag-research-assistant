# RAG Research Assistant

Hybrid retrieval (BM25 + FAISS) over research PDFs, with cross-encoder reranking,
metadata filtering and source-cited answers, served over FastAPI and packaged in
a Docker image that answers a question on first boot.

The design goal was not "wire a vector DB to an LLM". It was to build the parts
that are easy to get subtly wrong — chunking, fusion, reranking, citations,
evaluation — in a way that is **verifiable without a GPU, a model download or an
API key**, and then put the heavy models behind the same interfaces.

---

## Contents

- [What it does](#what-it-does)
- [Why it is built this way](#why-it-is-built-this-way)
- [Architecture](#architecture)
- [Quickstart (no models)](#quickstart-no-models)
- [With real models](#with-real-models)
- [Docker](#docker)
- [Evaluation](#evaluation)
- [API](#api)
- [Metadata filters](#metadata-filters)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Honest limitations](#honest-limitations)
- [License](#license)

---

## What it does

- **Ingests PDFs, `.txt` and `.md`** — extracts per-page text, records source,
  title, page numbers and any front-matter metadata it can find, then splits
  into overlapping word-window chunks along sentence boundaries.
- **Retrieves with two independent scorers** — Okapi BM25 (pure Python) and a
  dense nearest-neighbour search (FAISS `IndexFlatIP`, NumPy brute force when
  FAISS is absent), fused with Reciprocal Rank Fusion.
- **Reranks** the candidate list with a cross-encoder, so the passages that get
  quoted are chosen by a model that reads query and passage together.
- **Filters** by metadata (`year >= 2020`, `doc_id in [a, b]`, `tags contains
  "x"`, …) *before* fusion, so a filtered search is not a filtered-afterthought.
- **Answers with citations** — every sentence in an answer is traced back to the
  chunk it came from, and each citation resolves to a doc, page and snippet.
- **Scores itself** — Recall@K, MRR, nDCG@K and MAP against a labelled test set,
  with the single-retriever baselines run over the same queries.
- **Reports which backends are actually live** — `/health` and `/stats` name the
  embedder, vector index, reranker and fusion weights in use, so a result is never
  silently produced by the fallback path.

## Why it is built this way

Three decisions drive everything else:

**Every heavy dependency has a deterministic fallback.** `sentence-transformers`,
`faiss` and `torch` are optional. Without them the service runs a signed
feature-hashing embedder, a NumPy brute-force index and a lexical reranker. That
is what lets the whole test suite and CI run in seconds with no network, and it is
why a reviewer can clone this repo and see the pipeline work immediately. The
fallback is *not* presented as equivalent — `/health` says
`"embedder": "hashing-blake2b-512"` and the README says what that costs (see
[Honest limitations](#honest-limitations)).

**The fallbacks are lexical, and that has a measured consequence.** The proxy
embedder ranks largely by shared terms, which makes it strongly correlated with
BM25. Reciprocal Rank Fusion assumes its inputs contribute independent evidence;
at equal weight, two correlated rankings can outvote the better-calibrated sparse
one. So when the proxy embedder is live, its fusion weight is capped at
`RAG_LEXICAL_DENSE_WEIGHT` (default `0.5`). Measured on the labelled set, that cap
takes recall@1 from 0.786 → 0.810 and MRR from 0.925 → 0.937. A real bi-encoder
is uncorrelated with BM25 and keeps the full weight — the cap only ever applies to
the fallback.

**Labels are sentence anchors, not chunk ids.** A test label is an exact sentence
from a source document; the gold chunk is resolved at evaluation time by finding
which chunk contains that sentence. Re-chunk the corpus and the labels still
resolve, instead of silently pointing at the wrong text. `eval/testset.jsonl`
labels 42 queries against 20 documents (48 gold chunks), including 8 deliberate
distractor documents on adjacent topics.

## Architecture

```
                          ┌──────────────── ingest ────────────────┐
  PDF / .txt / .md  ──▶   │ extract text → normalise → sentence    │
                          │ split → word-window chunk (+overlap)   │
                          │ → embed → build BM25 + vector index    │
                          └────────────────────┬───────────────────┘
                                               │
                          data/index/  (chunks.jsonl · vectors.npy · meta.json)
                                               │
  question ────────────────────────────────────┼───────────────────────────────┐
                                               ▼                               │
                                      ┌─────────────────┐                      │
                                      │ metadata filter │  (rows allowed)      │
                                      └────────┬────────┘                      │
                        ┌──────────────────────┴──────────────────────┐        │
                        ▼                                             ▼        │
                 ┌────────────┐                               ┌─────────────┐  │
                 │ BM25 (top  │                               │ dense (top  │  │
                 │ candidate_k)│                              │ candidate_k)│  │
                 └─────┬──────┘                               └──────┬──────┘  │
                        └───────────────┬────────────────────────────┘         │
                                        ▼                                      │
                            ┌───────────────────────┐                          │
                            │ RRF fusion (+weights) │                          │
                            └───────────┬───────────┘                          │
                                        ▼                                      │
                            ┌───────────────────────┐                          │
                            │ cross-encoder rerank  │  (auto → lexical)        │
                            └───────────┬───────────┘                          │
                                        ▼                                      │
                     ┌──────────────────────────────────┐                      │
                     │ generator: extractive | openai   │                      │
                     │ → answer + per-sentence citations│                      │
                     └──────────────────────────────────┘                      │
                                        ▼                                      │
   FastAPI  /query · /search · /index/* · /evaluate · /stats · /health  ◀───────┘
```

Deeper notes on the retrieval and evaluation design are in
[`docs/architecture.md`](docs/architecture.md).

## Quickstart (no models)

Runs entirely offline with the NumPy + hashing + lexical fallbacks. Python 3.11+.

```bash
git clone https://github.com/iwdie/rag-research-assistant
cd rag-research-assistant
python -m venv .venv && . .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/bootstrap.py                        # generate corpus, build index, run one query
uvicorn app.main:app --reload                      # then open http://127.0.0.1:8000
```

`scripts/bootstrap.py` prints an answer with citations, so you can confirm the
whole path works before touching the API. The demo UI at `/` runs a query and
shows the answer, citations and the fused ranking; `/docs` is the OpenAPI console.

To score the pipeline against the labelled test set:

```bash
python scripts/evaluate.py --compare
```

## With real models

Install the heavier dependencies and the same interfaces switch to FAISS,
Sentence Transformers and the cross-encoder:

```bash
pip install -r requirements.txt -r requirements-vector.txt
python scripts/ingest.py --sample --rebuild        # rebuilds the index with the real embedder
uvicorn app.main:app --reload
```

The embedder and reranker are selected per-component (`auto` uses the model when
importable, otherwise the fallback). Force a path with `RAG_EMBEDDER_BACKEND` /
`RAG_RERANKER_BACKEND` set to `native` (fail loudly if the model is missing) or
`fallback` (never import the model).

> Rebuild the index after changing the embedder. A persisted index carries the
> embedding dimension it was built with; loading it under a different model is
> exactly the mismatch `IndexStore.load` rejects rather than answering nonsense.

## Docker

The image builds the index at build time *with the same embedder it will run
with*, and warms the model caches, so the container never needs the network at
runtime.

```bash
# ~250 MB: NumPy + hashing + lexical fallbacks, no model download
WITH_MODELS=false docker compose up --build

# ~2 GB: FAISS + Sentence Transformers + cross-encoder, baked into the image
docker compose up --build
```

```bash
docker run --rm -p 8000:8000 rag-research-assistant:latest
curl -s localhost:8000/health
```

The container runs as a non-root user, has a `HEALTHCHECK`, and deliberately runs
a single Uvicorn worker (the index lives in process memory — scale with replicas
behind a proxy, not workers).

## Evaluation

`scripts/evaluate.py --compare` runs the full hybrid pipeline and, over the same
42 labelled queries, each retriever alone. The numbers below are committed in
[`eval/baseline.txt`](eval/baseline.txt) and generated by the CI corpus, using the
dependency-free backends (25 chunks, 42 queries, 48 gold chunks):

| pipeline              | Recall@1 | Recall@3 | Recall@5 | MRR    | nDCG@5 |
| --------------------- | -------- | -------- | -------- | ------ | ------ |
| BM25 only             | 0.8095   | 0.9762   | 0.9762   | 0.9365 | 0.9345 |
| dense only            | 0.7262   | 0.9762   | 0.9881   | 0.8770 | 0.8952 |
| hybrid (RRF, no rerank)| 0.8095  | 0.9762   | 0.9881   | 0.9365 | 0.9389 |
| **hybrid + rerank**   | **0.8095** | **0.9762** | **0.9881** | **0.9365** | **0.9389** |

**Read this table honestly.** Hybrid retrieval beats or ties each single retriever
on every metric here, and on this corpus that is the real story: BM25 alone is
already strong (the documents use distinctive terminology), the dense proxy adds
recall@5 coverage, and fusion keeps the best of both without losing rank-1
precision. What the table does **not** show is a reranker win — at the shipped
neutral weight, hybrid + rerank is identical to hybrid without it. That is not an
accident to hide: with a lexical proxy embedder and a lexical fallback reranker
there is no non-lexical signal to add, so the reranking stage has nothing to
contribute on this corpus, and the default weight is set to the conservative
neutral value rather than a flattering one. The reranking stage pays off with the
real cross-encoder, which is exercised by the scheduled
[`models.yml`](.github/workflows/models.yml) workflow.

The evaluation harness is also available over HTTP (`POST /evaluate`), so the
numbers can be reproduced against a running service.

## API

| Method   | Path                     | Purpose                                              |
| -------- | ------------------------ | ---------------------------------------------------- |
| `GET`    | `/`                      | Demo UI                                              |
| `GET`    | `/docs`                  | OpenAPI / Swagger console                            |
| `GET`    | `/health`                | Liveness, corpus size, live backends                 |
| `GET`    | `/stats`                 | Index statistics, fusion weights, BM25 parameters    |
| `GET`    | `/documents`             | List indexed documents                               |
| `POST`   | `/query`                 | Answer a question, with citations                    |
| `POST`   | `/search`                | Retrieve passages only (no generation, no LLM cost)  |
| `POST`   | `/index/text`            | Index an inline document                             |
| `POST`   | `/index/documents`       | Upload PDF/`.txt`/`.md` files (multipart)            |
| `DELETE` | `/documents/{doc_id}`    | Remove a document and rebuild the indexes            |
| `POST`   | `/evaluate`              | Score retrieval against a labelled test set          |
| `GET`    | `/citations/{chunk_id}`  | Resolve a chunk id to a citable passage              |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' -d '{
  "question": "How does hybrid retrieval combine sparse and dense scores?",
  "top_k": 3,
  "filters": ["year >= 2020"],
  "debug": true
}'
```

```jsonc
{
  "question": "...",
  "answer": "Reciprocal rank fusion combines the two rankings ...",
  "citations": [{ "marker": 1, "chunk_id": "...", "title": "...", "page": 2, "snippet": "..." }],
  "generator": "extractive",
  "used_chunks": ["..."],
  "passages": [{ "chunk_id": "...", "score": 0.0328, "ranks": { "sparse": 1, "dense": 2 }, "...": "..." }],
  "debug": { "sparse_candidates": 50, "dense_candidates": 50, "fused": 97, "reranked": 5 }
}
```

```bash
# Upload a PDF and query it
curl -s -X POST localhost:8000/index/documents -F 'files=@paper.pdf'
curl -s -X POST localhost:8000/evaluate -H 'content-type: application/json' -d '{"compare": true}' | jq
```

Interactive docs: `http://127.0.0.1:8000/docs`.

## Metadata filters

Filters are parsed into a small, explicit grammar and applied before fusion.

```jsonc
// mapping form: equality (a list value becomes `in`)
{ "year": 2023, "doc_id": ["attention_2017", "bert_2019"] }

// clause form: field, operator, value
["year >= 2020", "tags contains \"retrieval\"", "source exists"]
```

Supported operators: `eq ne gt gte lt lte in nin contains exists` (with `= == != >
>= < <=` as aliases). String comparisons are case-insensitive; a filter on a list
field matches when **any** element matches. Unknown fields match nothing rather
than everything, so a typo fails closed.

## Configuration

All settings are `RAG_`-prefixed environment variables (or a local `.env`); see
[`.env.example`](.env.example) for the annotated list. The defaults run the whole
pipeline offline.

| Variable                       | Default                                  | Notes                                        |
| ------------------------------ | ---------------------------------------- | -------------------------------------------- |
| `RAG_INDEX_DIR`                | `data/index`                             | Persisted index location                     |
| `RAG_CHUNK_WORDS` / `..._OVERLAP_WORDS` | `180` / `40`                    | Word-window chunking                         |
| `RAG_EMBEDDER_BACKEND`         | `auto`                                   | `auto` \| `native` \| `fallback`             |
| `RAG_EMBEDDING_MODEL`          | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder when available                    |
| `RAG_USE_FAISS`                | `true`                                   | NumPy brute force when false/unavailable     |
| `RAG_FUSION`                   | `rrf`                                    | `rrf` \| `linear`                            |
| `RAG_DENSE_WEIGHT` / `RAG_SPARSE_WEIGHT` | `1.0` / `1.0`                  | Fusion weights (capped for the proxy embedder) |
| `RAG_LEXICAL_DENSE_WEIGHT`     | `0.5`                                    | Ceiling on the dense weight for the proxy    |
| `RAG_CANDIDATE_K` / `RAG_TOP_K`| `50` / `5`                               | Candidates per retriever / returned passages |
| `RAG_RERANK`                   | `true`                                   | Toggle the reranking stage                   |
| `RAG_RERANKER_BACKEND`         | `auto`                                   | `auto` \| `native` \| `fallback`             |
| `RAG_RERANKER_MODEL`           | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Cross-encoder when available                 |
| `RAG_GENERATOR`                | `extractive`                             | `extractive` \| `openai`                     |
| `RAG_OPENAI_BASE_URL`          | `https://api.openai.com/v1`              | Any OpenAI-compatible gateway                |

## Project layout

```
app/
  config.py              RAG_* settings (pydantic-settings)
  main.py                FastAPI app, lifespan, demo UI mount
  api/                   schemas + thin route handlers
  core/
    ingest.py            PDF/text extraction, Document model
    chunking.py          sentence splitting, word-window chunks
    embeddings.py        SentenceTransformers | hashing fallback
    bm25.py              Okapi BM25 (pure Python)
    vector_store.py      FAISS IndexFlatIP | NumPy brute force
    hybrid.py            fusion (RRF/linear), minmax, RetrievedChunk
    reranker.py          CrossEncoder | lexical fallback
    filters.py           metadata filter grammar
    retriever.py         filter → retrieve → fuse → rerank
    generator.py         extractive | OpenAI, citation building
  evaluation/metrics.py  Recall@K, MRR, nDCG@K, MAP, anchor resolution
  services/              IndexStore (persistence) + RAGAssistant (lifecycle)
  static/index.html      demo UI
scripts/                 make_sample_data · ingest · evaluate · bootstrap
data/sample/             20-document synthetic corpus (tracked)
eval/testset.jsonl       42 labelled queries (sentence anchors)
tests/                   pytest suite
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                  # 175 tests, no network, no models
ruff check app scripts tests
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) lints, **asserts the
sample corpus regenerates byte-identically**, runs the suite under coverage
(fail-under 80%), smoke-tests the CLI, and builds both Docker variants — then
verifies the container runs as a non-root user and serves `/health`.

## Honest limitations

- **The default backends are lexical, not semantic.** `hashing-blake2b-512` is
  signed feature hashing over unigrams and bigrams — deterministic, dependency-free
  and good for exercising the plumbing, but it does not understand paraphrases. The
  0.81 recall@1 above is a *lexical* result on a synthetic corpus whose documents
  use distinctive terminology. With a real bi-encoder the same corpus is an easier
  task, and the interesting numbers are the ones `models.yml` produces.
- **The reranker shows no gain under the fallback.** At the shipped neutral weight
  it is a no-op on this corpus, by design and by measurement — see
  [Evaluation](#evaluation). The cross-encoder is where reranking pays off.
- **The corpus is synthetic.** 20 documents summarising well-known papers, written
  to have controlled overlap and deliberate distractors. It is a correctness and
  regression harness, not a benchmark; do not quote these numbers as state of the
  art.
- **Single-process index.** The index lives in memory and is rebuilt/saved as one
  unit; scale out with replicas, not workers.
- **The extractive generator is deliberately dumb.** It quotes retrieved passages
  and refuses to answer when no passage shares a content term with the question.
  That is a feature for a retrieval demo (no hallucination, no cost); set
  `RAG_GENERATOR=openai` for a real synthesiser.

## License

MIT — see [LICENSE](LICENSE).
