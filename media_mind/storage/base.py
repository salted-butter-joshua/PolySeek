"""向量存储抽象层。

统一 Milvus Lite 与 Qdrant 两种后端，上层检索/摄取代码不感知具体实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SearchResult:
    """单条检索结果。"""

    id: str
    score: float  # 相似度（COSINE/IP，越大越相似，通常 0~1）
    file_path: str
    media_type: str  # image | video_frame | audio_transcript
    metadata: dict = field(default_factory=dict)  # frame_ts / segment_start 等动态字段


# 结构化字段（所有后端都建成标量字段，支持过滤）；其余进 metadata / dynamic field。
CORE_FIELDS = ("media_type", "file_path", "file_hash", "mtime")


@dataclass
class SearchFilter:
    """检索过滤条件（跨后端通用，各后端翻译成自己的表达式）。"""

    media_type: str | None = None
    mtime_min: float | None = None
    mtime_max: float | None = None

    def is_empty(self) -> bool:
        return (
            self.media_type is None
            and self.mtime_min is None
            and self.mtime_max is None
        )


class VectorStore(ABC):
    """向量数据库抽象基类。"""

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """创建 collection（幂等：已存在则跳过）。"""
        ...

    @abstractmethod
    def insert(self, vectors: np.ndarray, metadatas: list[dict]) -> list[str]:
        """批量插入向量 + 元数据，返回生成的 id 列表。"""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """向量检索 + 可选 metadata 过滤。"""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 id 删除。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回 collection 中的向量总数。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放连接/句柄。"""
        ...
