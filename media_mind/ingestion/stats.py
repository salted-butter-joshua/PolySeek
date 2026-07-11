"""离线嵌入耗时统计。

记录每种媒体类型（text / image / video / audio）的：
- 处理文件数、生成向量数
- 嵌入（编码）累计耗时 embed_seconds
- 预处理累计耗时 process_seconds（解码/抽帧/转写/切块）

用于前端展示"离线总嵌入时间（文本、图片、音视频分开）"。结果落盘到 JSON，
多次增量索引会累加。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

MEDIA_TYPES = ("text", "image", "video", "audio")


@dataclass
class TypeStat:
    files: int = 0
    vectors: int = 0
    embed_seconds: float = 0.0
    process_seconds: float = 0.0


@dataclass
class IngestionStats:
    per_type: dict[str, TypeStat] = field(
        default_factory=lambda: {t: TypeStat() for t in MEDIA_TYPES}
    )
    total_wall_seconds: float = 0.0
    updated_at: float = 0.0

    # ----------------------------------------------------------------- #
    def add_embed_time(self, media_type: str, seconds: float, vectors: int = 0) -> None:
        st = self.per_type.setdefault(media_type, TypeStat())
        st.embed_seconds += seconds
        st.vectors += vectors

    def add_process_time(self, media_type: str, seconds: float) -> None:
        self.per_type.setdefault(media_type, TypeStat()).process_seconds += seconds

    def add_files(self, media_type: str, n: int = 1) -> None:
        self.per_type.setdefault(media_type, TypeStat()).files += n

    @contextmanager
    def timer_embed(self, media_type: str, vectors: int = 0):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_embed_time(media_type, time.perf_counter() - t0, vectors)

    @contextmanager
    def timer_process(self, media_type: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_process_time(media_type, time.perf_counter() - t0)

    # ----------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "per_type": {k: asdict(v) for k, v in self.per_type.items()},
            "total_wall_seconds": self.total_wall_seconds,
            "updated_at": self.updated_at,
        }

    def save(self, path: str) -> None:
        self.updated_at = time.time()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> IngestionStats:
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        stats = cls(
            total_wall_seconds=raw.get("total_wall_seconds", 0.0),
            updated_at=raw.get("updated_at", 0.0),
        )
        for t, d in raw.get("per_type", {}).items():
            stats.per_type[t] = TypeStat(**d)
        return stats
