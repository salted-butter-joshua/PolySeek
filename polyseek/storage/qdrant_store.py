"""Qdrant 向量存储实现（Docker 服务，跨平台含 Windows）。

相比 Milvus Lite：
- 跨平台（Windows 原生可用），Docker 一键起，REST/gRPC 完善。
- payload 过滤能力强，适合以后横向扩展 / 云端托管。

metadata 全部存进 payload；结构化字段（media_type/mtime）建 payload 索引以加速过滤。
"""

from __future__ import annotations

import uuid

import numpy as np
from loguru import logger

from ..config import QdrantConfig
from .base import SearchFilter, SearchResult, VectorStore

_METRIC_UNSUPPORTED = "COSINE"  # Qdrant 默认对向量做归一化后按 Cosine 距离检索


class QdrantStore(VectorStore):
    def __init__(self, config: QdrantConfig):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("缺少 qdrant-client，请 pip install qdrant-client>=1.9") from e

        self.cfg = config
        self.collection_name = config.collection_name
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            grpc_port=config.grpc_port,
            prefer_grpc=config.prefer_grpc,
            api_key=config.api_key,
            timeout=config.timeout,  # 大批量索引时优化器构建 HNSW 会短暂阻塞写入，需更长超时
        )
        logger.info("Qdrant connected: {}:{} (timeout={}s)", config.host, config.port, config.timeout)

    # ------------------------------------------------------------------ #
    def create_collection(self, dimension: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection_name):
            logger.info("Collection '{}' exists, skip create.", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=self.cfg.hnsw_m, ef_construct=self.cfg.hnsw_ef_construct
                ),
            ),
        )
        # payload 索引：加速 media_type / mtime 过滤
        self.client.create_payload_index(
            self.collection_name, "media_type", field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.collection_name, "mtime", field_schema=models.PayloadSchemaType.FLOAT
        )
        logger.info(
            "Collection '{}' created. dim={}, distance=COSINE", self.collection_name, dimension
        )

    # ------------------------------------------------------------------ #
    def insert(self, vectors: np.ndarray, metadatas: list[dict]) -> list[str]:
        from qdrant_client import models

        if len(vectors) == 0:
            return []
        if len(vectors) != len(metadatas):
            raise ValueError(
                f"vectors ({len(vectors)}) 与 metadatas ({len(metadatas)}) 数量不一致"
            )

        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        points = [
            models.PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=metadatas[i])
            for i in range(len(vectors))
        ]
        self._upsert_with_retry(points)
        return ids

    def _upsert_with_retry(self, points, attempts: int = 4) -> None:
        """带指数退避的 upsert：优化器构建 HNSW 时可能短暂阻塞导致 DEADLINE_EXCEEDED，
        退避后重试即可恢复，避免整个大批量索引因偶发超时而中断。"""
        import time as _time

        for i in range(attempts):
            try:
                self.client.upsert(
                    collection_name=self.collection_name, points=points, wait=True
                )
                return
            except Exception as e:
                if i == attempts - 1:
                    raise
                delay = 2**i  # 1, 2, 4 秒
                logger.warning(
                    "Qdrant upsert 第 {}/{} 次失败（{}），{}s 后重试",
                    i + 1, attempts, type(e).__name__, delay,
                )
                _time.sleep(delay)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        # 未建 collection（尚未索引）时返回空，避免读服务直接 500
        if not self.client.collection_exists(self.collection_name):
            return []

        qfilter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        ).points

        out: list[SearchResult] = []
        for h in hits:
            payload = h.payload or {}
            out.append(
                SearchResult(
                    id=str(h.id),
                    score=float(h.score),
                    file_path=payload.get("file_path", ""),
                    media_type=payload.get("media_type", ""),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ("file_path", "media_type")
                    },
                )
            )
        return out

    def _build_filter(self, filters: SearchFilter | None):
        from qdrant_client import models

        if filters is None or filters.is_empty():
            return None

        must = []
        if filters.media_type:
            must.append(
                models.FieldCondition(
                    key="media_type", match=models.MatchValue(value=filters.media_type)
                )
            )
        if filters.mtime_min is not None or filters.mtime_max is not None:
            must.append(
                models.FieldCondition(
                    key="mtime",
                    range=models.Range(gte=filters.mtime_min, lte=filters.mtime_max),
                )
            )
        return models.Filter(must=must)

    def delete(self, ids: list[str]) -> None:
        from qdrant_client import models

        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=list(ids)),
            )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(self.collection_name, exact=True).count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
