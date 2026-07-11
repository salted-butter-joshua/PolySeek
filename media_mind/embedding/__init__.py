"""Embedding 后端工厂。

上层只依赖 :class:`EmbeddingService` 接口，具体后端由 config 决定，换模型不改代码。
后端的重依赖（cn_clip / transformers）都在各自模块内延迟导入，未选用的后端不会强制安装。
"""

from __future__ import annotations

from ..config import EmbeddingConfig
from .base import EmbeddingService, ImageInput

__all__ = ["EmbeddingService", "ImageInput", "create_embedding_service"]


def create_embedding_service(config: EmbeddingConfig) -> EmbeddingService:
    """按配置创建 Embedding 后端。"""
    backend = config.backend
    kwargs = {
        "model_name": config.model_name,
        "device": config.device,
        "cache_dir": config.model_cache_dir,
    }

    if backend == "chinese_clip":
        from .chinese_clip import ChineseClipEmbedding

        return ChineseClipEmbedding(**kwargs)
    if backend == "siglip":
        from .siglip import SigLIPEmbedding

        return SigLIPEmbedding(**kwargs)
    if backend == "openai_clip":
        from .openai_clip import OpenAIClipEmbedding

        return OpenAIClipEmbedding(**kwargs)

    raise ValueError(f"Unknown embedding backend: {backend!r}")
