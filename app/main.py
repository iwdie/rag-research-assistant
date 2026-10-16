"""FastAPI application: wiring, lifecycle and the small demo UI.

The index is built during application startup (via ``lifespan``) rather than at
import time. That matters because a module-level ``load()`` would download an
embedding model the moment anything imported the app - including the test
suite - and would make the import itself a side effect.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.services.assistant import RAGAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None, *, sample_dir: Path | None = SAMPLE_DIR) -> FastAPI:
    """Build the application.

    ``settings`` and ``sample_dir`` are injectable: tests point the index at a
    temporary directory and pass ``sample_dir=None`` to get a genuinely empty
    corpus, which is otherwise impossible to observe once a sample corpus ships
    with the repository.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = settings or get_settings()
        started = time.perf_counter()
        # An unreachable model host or a corrupt index must not stop the process
        # from serving /health - an empty-but-honest service is far easier to
        # debug than a container that exits on boot.
        try:
            app.state.assistant = RAGAssistant.load_or_build(active, sample_dir=sample_dir)
        except Exception as exc:
            logger.exception("failed to initialise the index: %s", exc)
            app.state.assistant = RAGAssistant.build(active)
        logger.info(
            "ready in %.2fs: %d documents, %d chunks, embedder=%s, vectors=%s",
            time.perf_counter() - started,
            len(app.state.assistant.store.documents),
            app.state.assistant.store.n_chunks,
            app.state.assistant.store.embedder.name,
            app.state.assistant.store.vectors.backend,
        )
        yield
        if active.persist_on_shutdown:
            try:
                app.state.assistant.save()
            except Exception as exc:
                logger.warning("could not persist the index on shutdown: %s", exc)

    app = FastAPI(
        title="RAG Research Assistant",
        version="1.0.0",
        description=(
            "Hybrid (BM25 + dense) retrieval over research PDFs with cross-encoder "
            "reranking and source-cited answers."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():  # pragma: no cover - only if the image omits static/
            return HTMLResponse("<h1>RAG Research Assistant</h1><p>See <a href='/docs'>/docs</a>.</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error", "type": type(exc).__name__})

    return app


app = create_app()
