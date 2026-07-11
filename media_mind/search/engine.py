"""统一检索引擎。

四种检索模式共用一个入口，区别只在于 query 的编码方式：
- 文搜图 / 文搜视频 / 文搜音频: text_search(media_type=...)
- 图搜图 / 图搜视频帧:          image_search(media_type=...)
- 全类型混合:                   hybrid_search()
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..embedding.base import EmbeddingService, ImageInput
from ..storage.base import SearchFilter, SearchResult, VectorStore


@dataclass
class TimedSearch:
    """带耗时的检索结果，供前端展示。"""

    results: list[SearchResult]
    embed_ms: float   # query 编码耗时（毫秒）
    search_ms: float  # 向量库检索耗时（毫秒）

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms


class SearchEngine:
    def __init__(self, embedding: EmbeddingService, store: VectorStore, hybrid_overfetch: int = 2):
        self.embed = embedding
        self.store = store
        self.hybrid_overfetch = max(1, hybrid_overfetch)

    # ------------------------------------------------------------------ #
    def text_search(
        self,
        query: str,
        top_k: int = 20,
        media_type: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> list[SearchResult]:
        """文本检索：文搜图 / 文搜视频帧 / 文搜音频转写。"""
        query_vector = self.embed.encode_text(query)
        return self._search(query_vector, top_k, media_type, time_range)

    def image_search(
        self,
        image: ImageInput,
        top_k: int = 20,
        media_type: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> list[SearchResult]:
        """图片检索：图搜图 / 图搜视频帧。"""
        query_vector = self.embed.encode_image(image)
        return self._search(query_vector, top_k, media_type, time_range)

    def _search(self, query_vector, top_k, media_type, time_range) -> list[SearchResult]:
        filters = SearchFilter(media_type=media_type)
        if time_range:
            filters.mtime_min, filters.mtime_max = time_range

        # 视频帧：同一视频多帧可能命中，超量取回后按视频去重。
        if media_type == "video_frame":
            raw = self.store.search(query_vector, top_k=top_k * 3, filters=filters)
            return self._dedup_videos(raw)[:top_k]

        return self.store.search(
            query_vector, top_k=top_k, filters=None if filters.is_empty() else filters
        )

    def hybrid_search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """混合搜索：同时搜图片/视频帧/音频转写，视频帧去重后统一按 score 排序。"""
        query_vector = self.embed.encode_text(query)
        raw = self.store.search(query_vector, top_k=top_k * self.hybrid_overfetch)

        out: list[SearchResult] = []
        seen_videos: set[str] = set()
        for r in raw:
            if r.media_type == "video_frame":
                if r.file_path in seen_videos:
                    continue
                seen_videos.add(r.file_path)
            out.append(r)
        return out[:top_k]

    # ------------------------------------------------------------------ #
    # 带耗时的检索（前端展示用）：分别计时 query 编码与向量检索。
    # ------------------------------------------------------------------ #
    def text_search_timed(
        self, query: str, top_k: int = 20, media_type: str | None = None
    ) -> TimedSearch:
        t0 = time.perf_counter()
        query_vector = self.embed.encode_text(query)
        t1 = time.perf_counter()
        results = self._search(query_vector, top_k, media_type, None)
        t2 = time.perf_counter()
        return TimedSearch(results, (t1 - t0) * 1000, (t2 - t1) * 1000)

    def image_search_timed(
        self, image: ImageInput, top_k: int = 20, media_type: str | None = None
    ) -> TimedSearch:
        t0 = time.perf_counter()
        query_vector = self.embed.encode_image(image)
        t1 = time.perf_counter()
        results = self._search(query_vector, top_k, media_type, None)
        t2 = time.perf_counter()
        return TimedSearch(results, (t1 - t0) * 1000, (t2 - t1) * 1000)

    def hybrid_search_timed(self, query: str, top_k: int = 20) -> TimedSearch:
        t0 = time.perf_counter()
        query_vector = self.embed.encode_text(query)
        t1 = time.perf_counter()
        raw = self.store.search(query_vector, top_k=top_k * self.hybrid_overfetch)
        t2 = time.perf_counter()
        out: list[SearchResult] = []
        seen: set[str] = set()
        for r in raw:
            if r.media_type == "video_frame":
                if r.file_path in seen:
                    continue
                seen.add(r.file_path)
            out.append(r)
        return TimedSearch(out[:top_k], (t1 - t0) * 1000, (t2 - t1) * 1000)

    @staticmethod
    def _dedup_videos(results: list[SearchResult]) -> list[SearchResult]:
        """同一视频只保留得分最高的帧，保留其 frame_ts 供 UI 定位。"""
        best: dict[str, SearchResult] = {}
        for r in results:
            cur = best.get(r.file_path)
            if cur is None or r.score > cur.score:
                best[r.file_path] = r
        return sorted(best.values(), key=lambda r: r.score, reverse=True)
