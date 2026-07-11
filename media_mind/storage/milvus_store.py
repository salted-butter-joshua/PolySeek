"""Milvus Lite 向量存储实现（嵌入式，无需 Docker）。

为什么选 Milvus Lite 而不是裸 FAISS：
1. FAISS 只是索引库，不管持久化、metadata 过滤、动态增删。
2. Milvus Lite 一个 ``.db`` 文件搞定，支持标量过滤，接口与分布式 Milvus 一致，
   数据量涨了可无缝迁移到 Milvus 集群。

索引选型：
- < 10 万：FLAT（暴力，recall 100%）
- 10~100 万：HNSW（M=16, ef=200, recall ~98%，快 10x）
- > 100 万：IVF_FLAT / IVF_PQ（省内存）

注意：Milvus Lite 目前仅支持 Linux / macOS。Windows 请用 Qdrant 后端或容器运行。
"""

from __future__ import annotations

import uuid

import numpy as np
from loguru import logger

from ..config import MilvusConfig
from .base import SearchFilter, SearchResult, VectorStore


class MilvusLiteStore(VectorStore):
    def __init__(self, config: MilvusConfig):
        try:
            from pymilvus import MilvusClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("缺少 pymilvus，请 pip install pymilvus>=2.4") from e

        self.cfg = config
        self.collection_name = config.collection_name
        self.index_type = config.index_type
        self.metric_type = config.metric_type
        self.client = MilvusClient(uri=config.db_path)
        logger.info("Milvus Lite connected: {}", config.db_path)

    # ------------------------------------------------------------------ #
    def create_collection(self, dimension: int) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.collection_name):
            logger.info("Collection '{}' exists, skip create.", self.collection_name)
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("media_type", DataType.VARCHAR, max_length=32)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("file_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("mtime", DataType.DOUBLE)

        index_params = self.client.prepare_index_params()
        if self.index_type == "HNSW":
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type=self.metric_type,
                params={"M": self.cfg.hnsw_m, "efConstruction": self.cfg.hnsw_ef_construction},
            )
        elif self.index_type == "IVF_FLAT":
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type=self.metric_type,
                params={"nlist": self.cfg.ivf_nlist},
            )
        else:  # FLAT
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type=self.metric_type,
            )
        # 标量字段索引，加速过滤
        index_params.add_index(field_name="media_type", index_type="INVERTED")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "Collection '{}' created. dim={}, index={}, metric={}",
            self.collection_name, dimension, self.index_type, self.metric_type,
        )

    # ------------------------------------------------------------------ #
    def insert(self, vectors: np.ndarray, metadatas: list[dict]) -> list[str]:
        if len(vectors) == 0:
            return []
        if len(vectors) != len(metadatas):
            raise ValueError(
                f"vectors ({len(vectors)}) 与 metadatas ({len(metadatas)}) 数量不一致"
            )

        # id 用 UUID：增量场景通过 file_hash/state DB 定位记录，UUID 避免 id 冲突。
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        data = [
            {"id": ids[i], "vector": vectors[i].tolist(), **metadatas[i]}
            for i in range(len(vectors))
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        # 未建 collection（尚未索引）时返回空，避免读服务直接报错
        if not self.client.has_collection(self.collection_name):
            return []

        filter_expr = self._build_filter_expr(filters)

        params: dict = {}
        if self.index_type == "HNSW":
            params = {"ef": max(top_k * 2, 64)}
        elif self.index_type == "IVF_FLAT":
            params = {"nprobe": 16}

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector.tolist()],
            limit=top_k,
            output_fields=["media_type", "file_path", "mtime", "file_hash"],
            search_params={"params": params},
            filter=filter_expr or "",
        )

        out: list[SearchResult] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            core = {"media_type", "file_path", "mtime", "file_hash"}
            out.append(
                SearchResult(
                    id=str(hit["id"]),
                    score=float(hit["distance"]),
                    file_path=entity.get("file_path", ""),
                    media_type=entity.get("media_type", ""),
                    metadata={
                        "mtime": entity.get("mtime"),
                        "file_hash": entity.get("file_hash"),
                        **{k: v for k, v in entity.items() if k not in core},
                    },
                )
            )
        return out

    @staticmethod
    def _build_filter_expr(filters: SearchFilter | None) -> str:
        if filters is None or filters.is_empty():
            return ""
        conds: list[str] = []
        if filters.media_type:
            conds.append(f'media_type == "{filters.media_type}"')
        if filters.mtime_min is not None:
            conds.append(f"mtime >= {filters.mtime_min}")
        if filters.mtime_max is not None:
            conds.append(f"mtime <= {filters.mtime_max}")
        return " and ".join(conds)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(collection_name=self.collection_name, ids=ids)

    def count(self) -> int:
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
