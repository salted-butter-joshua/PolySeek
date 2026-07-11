"""CLI 入口（Typer + Rich）。

命令：
    polyseek index [--full]      建立/更新索引
    polyseek search <query>      文本搜索（可 --type 限定媒体类型）
    polyseek similar <image>     以图搜图
    polyseek stats               查看索引统计
    polyseek serve               启动 REST API
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..config import load_config
from ..context import build_pipeline, build_search_context
from ..logging_setup import setup_logging

app = typer.Typer(help="PolySeek：自托管多模态语义检索引擎", no_args_is_help=True)
console = Console()

ConfigOpt = typer.Option("config.yaml", "--config", "-c", help="配置文件路径")


@app.command()
def index(
    full: bool = typer.Option(False, "--full", help="全量重建索引（否则增量）"),
    config: str = ConfigOpt,
):
    """建立或更新索引。"""
    cfg = load_config(config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg)
    pipeline = build_pipeline(ctx)
    try:
        if full:
            console.print("[bold yellow]Running FULL index ...[/bold yellow]")
            pipeline.run_full_index(cfg.data_sources)
        else:
            console.print("[bold green]Running incremental index ...[/bold green]")
            pipeline.run_incremental_index(cfg.data_sources)
    finally:
        ctx.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索文本"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    media_type: str = typer.Option(
        None, "--type", "-t", help="image | video_frame | audio_transcript"
    ),
    config: str = ConfigOpt,
):
    """文本搜索（不指定 --type 则跨全类型混合搜索）。"""
    cfg = load_config(config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg, ensure_collection=False)
    try:
        if media_type:
            results = ctx.engine.text_search(query, top_k=top_k, media_type=media_type)
        else:
            results = ctx.engine.hybrid_search(query, top_k=top_k)
        _display_results(results)
    finally:
        ctx.close()


@app.command()
def similar(
    image_path: str = typer.Argument(..., help="查询图片路径"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    config: str = ConfigOpt,
):
    """以图搜图。"""
    cfg = load_config(config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg, ensure_collection=False)
    try:
        results = ctx.engine.image_search(image_path, top_k=top_k, media_type="image")
        _display_results(results)
    finally:
        ctx.close()


@app.command()
def stats(config: str = ConfigOpt):
    """查看索引统计信息。"""
    cfg = load_config(config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg, ensure_collection=False)
    try:
        table = Table(title="Index Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total vectors", str(ctx.store.count()))
        table.add_row("Embedding backend", cfg.embedding.backend)
        table.add_row("Embedding model", cfg.embedding.model_name)
        table.add_row("Vector dimension", str(ctx.embedding.dimension))
        table.add_row("Vector store", cfg.vector_store.backend)
        console.print(table)
    finally:
        ctx.close()


@app.command()
def serve(config: str = ConfigOpt):
    """启动 REST API 服务。"""
    import uvicorn

    cfg = load_config(config)
    setup_logging(cfg.logging)
    # 通过环境变量把配置路径传给 API 进程
    import os

    os.environ["POLYSEEK_CONFIG"] = config
    uvicorn.run(
        "polyseek.api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        log_level=cfg.logging.level.lower(),
    )


def _display_results(results) -> None:
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Type", style="cyan", width=16)
    table.add_column("File", style="white")
    table.add_column("Extra", style="dim")

    for i, r in enumerate(results, 1):
        extra = ""
        if r.media_type == "video_frame":
            extra = f"@ {r.metadata.get('frame_ts', 0):.1f}s"
        elif r.media_type == "audio_transcript":
            start = r.metadata.get("segment_start", 0)
            text = str(r.metadata.get("transcript_text", ""))[:40]
            extra = f"@ {start:.1f}s | {text}..."
        table.add_row(
            str(i), f"{r.score:.4f}", r.media_type, Path(r.file_path).name, extra
        )
    console.print(table)


if __name__ == "__main__":
    app()
