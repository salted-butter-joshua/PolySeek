"""API 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResultItem(BaseModel):
    id: str
    score: float
    file_path: str
    media_type: str
    metadata: dict = Field(default_factory=dict)


class Timing(BaseModel):
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


class SearchResponse(BaseModel):
    query: str | None = None
    count: int
    results: list[ResultItem]
    timing: Timing = Field(default_factory=Timing)


class TypeTiming(BaseModel):
    files: int
    vectors: int
    embed_seconds: float
    process_seconds: float
    vectors_per_second: float = 0.0


class TimingReport(BaseModel):
    """离线嵌入耗时（分类型）。"""

    per_type: dict[str, TypeTiming]
    total_wall_seconds: float
    updated_at: float


class EffectiveConfig(BaseModel):
    """当前生效的关键配置（供前端展示/编辑）。"""

    embedding_backend: str
    embedding_model: str
    device: str
    batch_size: int
    vector_store: str
    dimension: int


class StatsResponse(BaseModel):
    total_vectors: int
    embedding_backend: str
    embedding_model: str
    dimension: int
    vector_store: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
