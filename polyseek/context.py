"""应用组装（依赖装配）。

集中一处把 config → embedding → store → engine/pipeline 装配好，
供 CLI / API / WebUI 复用，避免各入口重复 wiring。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, ensure_data_dirs
from .embedding import EmbeddingService, create_embedding_service
from .ingestion import FileScanner, IngestionPipeline
from .search import SearchEngine
from .storage import VectorStore, create_vector_store


@dataclass
class AppContext:
    config: AppConfig
    embedding: EmbeddingService
    store: VectorStore
    engine: SearchEngine

    def close(self) -> None:
        self.store.close()


def build_search_context(config: AppConfig, ensure_collection: bool = True) -> AppContext:
    """装配检索所需组件（懒装配 embedding + store + engine）。"""
    ensure_data_dirs(config)
    embedding = create_embedding_service(config.embedding)
    store = create_vector_store(config.vector_store)
    if ensure_collection:
        store.create_collection(embedding.dimension)
    engine = SearchEngine(embedding, store, hybrid_overfetch=config.search.hybrid_overfetch)
    return AppContext(config=config, embedding=embedding, store=store, engine=engine)


def build_pipeline(context: AppContext) -> IngestionPipeline:
    """基于已装配的 context 构建摄取管道。"""
    scanner = FileScanner(context.config.indexing)
    return IngestionPipeline(context.embedding, context.store, scanner, context.config)
