"""FastAPI REST API。

支持文搜图/视频/音频、以图搜图、统计、健康检查。
使用现代 lifespan（替代已弃用的 on_event），启动时装配一次 SearchEngine。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .. import __version__
from ..config import load_config
from ..context import AppContext, build_search_context
from ..ingestion.stats import IngestionStats
from ..logging_setup import setup_logging
from ..search.engine import TimedSearch
from ..storage.base import SearchResult
from .schemas import (
    EffectiveConfig,
    HealthResponse,
    ResultItem,
    SearchResponse,
    StatsResponse,
    Timing,
    TimingReport,
    TypeTiming,
)

_ctx: AppContext | None = None


def _require_ctx() -> AppContext:
    if _ctx is None:  # pragma: no cover - lifespan 保证已初始化
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ctx
    config_path = os.environ.get("MEDIA_MIND_CONFIG", "config.yaml")
    cfg = load_config(config_path)
    setup_logging(cfg.logging)
    _ctx = build_search_context(cfg, ensure_collection=False)
    app.state.serve_files = cfg.api.serve_files
    try:
        yield
    finally:
        _ctx.close()


def create_app() -> FastAPI:
    app = FastAPI(title="media-mind API", version=__version__, lifespan=lifespan)

    # CORS（origins 在 lifespan 前无法读 config，这里放开由反代收敛；如需精细控制可改）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _to_items(results: list[SearchResult]) -> list[ResultItem]:
        return [
            ResultItem(
                id=r.id,
                score=r.score,
                file_path=r.file_path,
                media_type=r.media_type,
                metadata=r.metadata,
            )
            for r in results
        ]

    def _to_response(query: str | None, ts: TimedSearch) -> SearchResponse:
        return SearchResponse(
            query=query,
            count=len(ts.results),
            results=_to_items(ts.results),
            timing=Timing(
                embed_ms=ts.embed_ms, search_ms=ts.search_ms, total_ms=ts.total_ms
            ),
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(version=__version__)

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats():
        ctx = _require_ctx()
        return StatsResponse(
            total_vectors=ctx.store.count(),
            embedding_backend=ctx.config.embedding.backend,
            embedding_model=ctx.config.embedding.model_name,
            dimension=ctx.embedding.dimension,
            vector_store=ctx.config.vector_store.backend,
        )

    @app.get("/api/search/text", response_model=SearchResponse)
    async def text_search(
        q: str = Query(..., description="搜索文本；media_type=text 即文搜文"),
        top_k: int = Query(20, ge=1, le=100),
        media_type: str | None = Query(None),
    ):
        ctx = _require_ctx()
        if media_type:
            ts = ctx.engine.text_search_timed(q, top_k=top_k, media_type=media_type)
        else:
            ts = ctx.engine.hybrid_search_timed(q, top_k=top_k)
        return _to_response(q, ts)

    @app.post("/api/search/image", response_model=SearchResponse)
    async def image_search(
        file: UploadFile = File(...),
        top_k: int = Query(20, ge=1, le=100),
        media_type: str | None = Query(None),
    ):
        ctx = _require_ctx()
        suffix = Path(file.filename or "q.jpg").suffix or ".jpg"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            ts = ctx.engine.image_search_timed(
                tmp_path, top_k=top_k, media_type=media_type or "image"
            )
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        return _to_response(None, ts)

    @app.get("/api/config", response_model=EffectiveConfig)
    async def get_config():
        ctx = _require_ctx()
        return EffectiveConfig(
            embedding_backend=ctx.config.embedding.backend,
            embedding_model=ctx.config.embedding.model_name,
            device=ctx.config.embedding.device,
            batch_size=ctx.config.embedding.batch_size,
            vector_store=ctx.config.vector_store.backend,
            dimension=ctx.embedding.dimension,
        )

    @app.get("/api/timing", response_model=TimingReport)
    async def get_timing():
        """离线嵌入耗时（文本/图片/视频/音频分开）。"""
        ctx = _require_ctx()
        stats = IngestionStats.load(ctx.config.indexing.stats_path)
        per_type = {}
        for t, st in stats.per_type.items():
            vps = st.vectors / st.embed_seconds if st.embed_seconds > 0 else 0.0
            per_type[t] = TypeTiming(
                files=st.files,
                vectors=st.vectors,
                embed_seconds=st.embed_seconds,
                process_seconds=st.process_seconds,
                vectors_per_second=vps,
            )
        return TimingReport(
            per_type=per_type,
            total_wall_seconds=stats.total_wall_seconds,
            updated_at=stats.updated_at,
        )

    @app.get("/api/file")
    async def serve_file(path: str = Query(...)):
        if not app.state.serve_files:
            raise HTTPException(status_code=403, detail="File serving disabled")
        file_path = Path(path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    return app


app = create_app()
