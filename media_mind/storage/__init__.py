"""向量存储工厂。

上层只依赖 :class:`VectorStore` 接口，具体后端由 config 决定。
"""

from __future__ import annotations

from ..config import VectorStoreConfig
from .base import SearchFilter, SearchResult, VectorStore

__all__ = [
    "VectorStore",
    "SearchResult",
    "SearchFilter",
    "create_vector_store",
]


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """按配置创建向量存储后端。"""
    if config.backend == "milvus_lite":
        from .milvus_store import MilvusLiteStore

        return MilvusLiteStore(config.milvus)
    if config.backend == "qdrant":
        from .qdrant_store import QdrantStore

        return QdrantStore(config.qdrant)

    raise ValueError(f"Unknown vector store backend: {config.backend!r}")
