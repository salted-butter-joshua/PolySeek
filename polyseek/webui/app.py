"""Gradio Web UI（增强版）。

功能：
- 前端切换配置：Embedding 后端 / 模型 / 向量库 / 设备 / batch_size / top_k，并热重载引擎
- 四种检索：文搜文、文搜图、图搜图、混合；实时展示 query 编码耗时 / 向量检索耗时 / 总耗时
- 离线嵌入统计面板：文本/图片/视频/音频分开的嵌入总时长、文件数、向量数、吞吐

启动：python -m polyseek.webui.app
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from ..config import AppConfig, load_config
from ..context import AppContext, build_search_context
from ..ingestion.stats import IngestionStats
from ..logging_setup import setup_logging
from ..search.engine import TimedSearch
from ..storage.base import SearchResult

IMAGE_TYPES = {"image", "video_frame"}

# 运行时可热重载的状态
_STATE: dict = {"ctx": None, "base_config": None}


# --------------------------------------------------------------------------- #
# 引擎装配 / 热重载
# --------------------------------------------------------------------------- #
def _rebuild(overrides: dict) -> str:
    base: AppConfig = _STATE["base_config"]
    data = base.model_dump()
    data["embedding"]["backend"] = overrides["backend"]
    data["embedding"]["model_name"] = overrides["model_name"]
    data["embedding"]["device"] = overrides["device"]
    data["embedding"]["batch_size"] = int(overrides["batch_size"])
    data["vector_store"]["backend"] = overrides["vector_store"]

    try:
        cfg = AppConfig(**data)
        old: AppContext | None = _STATE["ctx"]
        ctx = build_search_context(cfg, ensure_collection=False)
        _STATE["ctx"] = ctx
        if old is not None:
            old.close()
        return (
            f"✅ 已重载：backend={cfg.embedding.backend}, model={cfg.embedding.model_name}, "
            f"store={cfg.vector_store.backend}, device={cfg.embedding.device}, "
            f"dim={ctx.embedding.dimension}"
        )
    except Exception as e:
        logger.exception("Reload failed")
        return f"❌ 重载失败：{e}"


def _ctx() -> AppContext:
    if _STATE["ctx"] is None:
        _STATE["ctx"] = build_search_context(_STATE["base_config"], ensure_collection=False)
    return _STATE["ctx"]


# --------------------------------------------------------------------------- #
# 结果格式化
# --------------------------------------------------------------------------- #
def _rows(results: list[SearchResult]) -> list[list]:
    rows = []
    for r in results:
        extra = ""
        if r.media_type == "video_frame":
            extra = f"@ {r.metadata.get('frame_ts', 0):.1f}s"
        elif r.media_type == "audio_transcript":
            extra = str(r.metadata.get("transcript_text", ""))[:60]
        elif r.media_type == "text":
            extra = str(r.metadata.get("chunk_text", ""))[:60]
        rows.append([f"{r.score:.4f}", r.media_type, Path(r.file_path).name, extra])
    return rows


def _gallery(results: list[SearchResult]) -> list:
    items = []
    for r in results:
        if r.media_type in IMAGE_TYPES and Path(r.file_path).is_file():
            cap = f"{r.score:.3f}"
            if r.media_type == "video_frame":
                cap += f" @ {r.metadata.get('frame_ts', 0):.1f}s"
            items.append((r.file_path, cap))
    return items


def _timing_md(ts: TimedSearch) -> str:
    return (
        f"**query 编码**：{ts.embed_ms:.2f} ms　|　"
        f"**向量检索**：{ts.search_ms:.2f} ms　|　"
        f"**总耗时**：{ts.total_ms:.2f} ms"
    )


# --------------------------------------------------------------------------- #
# 检索回调
# --------------------------------------------------------------------------- #
def do_text_search(query, media_type, top_k):
    if not query:
        return [], [], ""
    engine = _ctx().engine
    mt = None if media_type == "all" else media_type
    if mt:
        ts = engine.text_search_timed(query, top_k=int(top_k), media_type=mt)
    else:
        ts = engine.hybrid_search_timed(query, top_k=int(top_k))
    return _gallery(ts.results), _rows(ts.results), _timing_md(ts)


def do_image_search(image_path, media_type, top_k):
    if not image_path:
        return [], [], ""
    engine = _ctx().engine
    mt = "image" if media_type == "all" else media_type
    ts = engine.image_search_timed(image_path, top_k=int(top_k), media_type=mt)
    return _gallery(ts.results), _rows(ts.results), _timing_md(ts)


def load_timing():
    ctx = _ctx()
    stats = IngestionStats.load(ctx.config.indexing.stats_path)
    rows = []
    for t in ("text", "image", "video", "audio"):
        st = stats.per_type.get(t)
        if st is None:
            continue
        vps = st.vectors / st.embed_seconds if st.embed_seconds > 0 else 0.0
        rows.append([
            t, st.files, st.vectors,
            f"{st.embed_seconds:.2f}", f"{st.process_seconds:.2f}", f"{vps:.1f}",
        ])
    summary = (
        f"**离线索引总墙钟时间**：{stats.total_wall_seconds:.1f} s　|　"
        f"**向量总数**：{sum(s.vectors for s in stats.per_type.values())}"
    )
    return rows, summary


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui(config_path: str = "config.yaml"):
    import gradio as gr

    cfg = load_config(config_path)
    setup_logging(cfg.logging)
    _STATE["base_config"] = cfg

    with gr.Blocks(title="polyseek") as demo:
        gr.Markdown("# 🧠 polyseek — NAS 多模态检索\n文搜文 · 文搜图 · 图搜图 · 混合检索")

        # ---------------- 配置面板 ----------------
        with gr.Accordion("⚙️ 运行配置（可热重载引擎）", open=False):
            with gr.Row():
                backend = gr.Dropdown(
                    ["chinese_clip", "siglip", "openai_clip"],
                    value=cfg.embedding.backend, label="Embedding 后端",
                )
                model_name = gr.Textbox(value=cfg.embedding.model_name, label="模型名")
                store = gr.Dropdown(
                    ["milvus_lite", "qdrant"], value=cfg.vector_store.backend,
                    label="向量库",
                )
            with gr.Row():
                device = gr.Dropdown(["cpu", "cuda", "mps"], value=cfg.embedding.device, label="设备")
                batch = gr.Number(value=cfg.embedding.batch_size, label="batch_size", precision=0)
                apply_btn = gr.Button("应用并重载引擎", variant="primary")
            reload_status = gr.Markdown()
            apply_btn.click(
                lambda b, m, s, d, bs: _rebuild(
                    {"backend": b, "model_name": m, "vector_store": s, "device": d, "batch_size": bs}
                ),
                [backend, model_name, store, device, batch],
                reload_status,
            )
            gr.Markdown(
                "> 注意：切换后端/模型会改变向量维度，需与已建索引维度一致，否则检索会报错。"
            )

        top_k = gr.Slider(1, 50, value=12, step=1, label="Top-K")

        # ---------------- 文本检索 ----------------
        with gr.Tab("文本检索（文搜文/图/视频/音频）"):
            with gr.Row():
                txt = gr.Textbox(label="查询文本", placeholder="红色的圆形 / 海边日落", scale=4)
                mt1 = gr.Dropdown(
                    ["all", "text", "image", "video_frame", "audio_transcript"],
                    value="image", label="目标类型",
                )
            btn1 = gr.Button("搜索", variant="primary")
            timing1 = gr.Markdown()
            gal1 = gr.Gallery(label="结果", columns=4, height="auto")
            tbl1 = gr.Dataframe(headers=["Score", "Type", "File", "Extra"], label="明细")
            btn1.click(do_text_search, [txt, mt1, top_k], [gal1, tbl1, timing1])

        # ---------------- 以图搜图 ----------------
        with gr.Tab("以图搜图"):
            with gr.Row():
                img = gr.Image(type="filepath", label="查询图片")
                with gr.Column():
                    mt2 = gr.Dropdown(["all", "image", "video_frame"], value="image", label="目标类型")
                    btn2 = gr.Button("搜索", variant="primary")
            timing2 = gr.Markdown()
            gal2 = gr.Gallery(label="结果", columns=4, height="auto")
            tbl2 = gr.Dataframe(headers=["Score", "Type", "File", "Extra"], label="明细")
            btn2.click(do_image_search, [img, mt2, top_k], [gal2, tbl2, timing2])

        # ---------------- 离线嵌入统计 ----------------
        with gr.Tab("📊 离线嵌入统计"):
            gr.Markdown("离线索引阶段各媒体类型的嵌入耗时（文本 / 图片 / 视频 / 音频分开）")
            refresh = gr.Button("刷新统计")
            timing_summary = gr.Markdown()
            timing_tbl = gr.Dataframe(
                headers=["类型", "文件数", "向量数", "嵌入秒", "预处理秒", "向量/秒"],
                label="分类型嵌入耗时",
            )
            refresh.click(load_timing, None, [timing_tbl, timing_summary])
            demo.load(load_timing, None, [timing_tbl, timing_summary])

    return demo


def main():
    config_path = os.environ.get("POLYSEEK_CONFIG", "config.yaml")
    demo = build_ui(config_path)
    # Gradio 4 默认只允许服务 CWD/临时目录下的文件；媒体在数据源目录（如 /data/media/*），
    # 需显式加入 allowed_paths，否则结果画廊里的图片无法显示（InvalidPathError）。
    cfg = _STATE["base_config"]
    allowed = []
    for ds in cfg.data_sources:
        p = Path(ds.path)
        allowed.append(str(p if p.is_absolute() else p.resolve()))
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=allowed)


if __name__ == "__main__":
    main()
