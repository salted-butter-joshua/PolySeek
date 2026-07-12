"""强类型配置模型与加载器。

设计要点：
- 所有配置字段都有 pydantic 模型，加载 YAML 时立即校验，非法配置在启动阶段
  就报错，而不是等到运行中途才崩。
- 支持环境变量覆盖（前缀 ``POLYSEEK__``，双下划线表示嵌套），方便容器化部署。
- 上层模块只依赖强类型对象（``AppConfig`` 及其子模型），不再传裸 dict，IDE 有补全，
  重构安全。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# ---------------------------------------------------------------------------
# 子模型
# ---------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    backend: Literal["chinese_clip", "siglip", "openai_clip"] = "chinese_clip"
    model_name: str = "ViT-B-16"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = Field(32, ge=1)
    model_cache_dir: str = "./models"
    dimension: int | None = None  # 由模型决定，通常自动探测


class MilvusConfig(BaseModel):
    db_path: str = "./data/milvus.db"
    collection_name: str = "media_embeddings"
    index_type: Literal["FLAT", "IVF_FLAT", "HNSW"] = "HNSW"
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"
    hnsw_m: int = Field(16, ge=4, le=64)
    hnsw_ef_construction: int = Field(200, ge=8)
    ivf_nlist: int = Field(128, ge=1)


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    api_key: str | None = None
    collection_name: str = "media_embeddings"
    hnsw_m: int = Field(16, ge=4, le=64)
    hnsw_ef_construct: int = Field(200, ge=8)
    timeout: int = Field(120, ge=1)  # gRPC/REST 超时秒数（大批量索引时优化器阻塞需更长）


class VectorStoreConfig(BaseModel):
    backend: Literal["milvus_lite", "qdrant"] = "milvus_lite"
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


class DataSource(BaseModel):
    path: str
    media_types: list[Literal["image", "video", "audio", "text"]] = Field(
        default_factory=lambda: ["image", "video", "audio", "text"]
    )
    recursive: bool = True


class ImageConfig(BaseModel):
    supported_formats: list[str] = Field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff",
        ]
    )
    max_dimension: int = 1024


class VideoConfig(BaseModel):
    supported_formats: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".mkv", ".avi", ".webm"]
    )
    frame_interval_seconds: float = Field(2.0, gt=0)
    max_frames_per_video: int = Field(500, ge=1)
    thumbnail_size: tuple[int, int] = (336, 336)
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


class AudioConfig(BaseModel):
    supported_formats: list[str] = Field(
        default_factory=lambda: [".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac"]
    )
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "base"
    language: str = "zh"
    min_segment_chars: int = 2


class TextConfig(BaseModel):
    """纯文本文档（用于文搜文）。"""

    supported_formats: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".markdown"]
    )
    chunk_chars: int = Field(200, ge=16)  # 每个文本块最大字符数（CLIP 文本编码有 token 上限）
    chunk_overlap: int = Field(20, ge=0)  # 相邻块的重叠字符数
    min_chunk_chars: int = 4
    encoding: str = "utf-8"


class IndexingConfig(BaseModel):
    state_db_path: str = "./data/index_state.db"
    stats_path: str = "./data/index_stats.json"  # 离线嵌入耗时统计落盘
    hash_algorithm: Literal["xxhash", "md5", "sha256"] = "xxhash"
    large_file_threshold_mb: int = 100
    workers: int = Field(4, ge=1)


class SearchConfig(BaseModel):
    default_top_k: int = Field(20, ge=1)
    hybrid_overfetch: int = Field(2, ge=1)


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8900
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    serve_files: bool = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = None


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------


# 当前正在加载的 YAML 路径（load_config 设置，供 settings source 读取）。
# 优先级：环境变量 > YAML 文件 > 默认值。
_active_yaml_path: str | Path | None = None


class AppConfig(BaseSettings):
    """应用全局配置。

    优先级：环境变量 > YAML 文件 > 字段默认值。
    环境变量覆盖示例：``POLYSEEK__EMBEDDING__DEVICE=cuda``。
    """

    model_config = SettingsConfigDict(
        env_prefix="POLYSEEK__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 顺序即优先级（前者覆盖后者）：init > env > yaml。
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if _active_yaml_path is not None and Path(_active_yaml_path).exists():
            # 显式 UTF-8：否则 Windows 上会用 GBK 解码含中文的 YAML 而报错。
            sources.append(
                YamlConfigSettingsSource(
                    settings_cls,
                    yaml_file=_active_yaml_path,
                    yaml_file_encoding="utf-8",
                )
            )
        return tuple(sources)

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    data_sources: list[DataSource] = Field(default_factory=list)
    image: ImageConfig = Field(default_factory=ImageConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("data_sources")
    @classmethod
    def _warn_empty_sources(cls, v: list[DataSource]) -> list[DataSource]:
        # 允许为空（例如只跑 search / API），此处不强制，仅保持接口一致。
        return v


# ---------------------------------------------------------------------------
# 加载器
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """从 YAML 文件加载配置并做校验；环境变量优先级高于文件。

    Args:
        config_path: YAML 配置文件路径。文件不存在时使用纯默认值 + 环境变量。

    Returns:
        校验通过的 :class:`AppConfig` 实例。
    """
    global _active_yaml_path
    _active_yaml_path = config_path
    try:
        # YAML 通过自定义 settings source 加载，优先级低于环境变量。
        return AppConfig()
    finally:
        _active_yaml_path = None


def ensure_data_dirs(config: AppConfig) -> None:
    """确保运行所需的数据/模型目录存在（幂等）。"""
    for p in (
        Path(config.embedding.model_cache_dir),
        Path(config.vector_store.milvus.db_path).parent,
        Path(config.indexing.state_db_path).parent,
    ):
        p.mkdir(parents=True, exist_ok=True)
