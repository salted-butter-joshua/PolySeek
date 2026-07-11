"""文件系统扫描器 + 增量检测。

增量策略：SQLite（index_state.db）记录每个已索引文件的 hash / mtime / 对应向量 id。
扫描时对比：
- 新文件（path 不在 DB）           → 新建索引
- mtime 变且 hash 变              → 重新索引（先删旧向量）
- DB 有但磁盘没了                  → 从向量库删除
- 未变化                          → 跳过

为什么用 xxhash：对大文件 hash 比 md5 快 5-10x，且这里只需检测"是否变化"，不需要
密码学安全性。超大文件（默认 >100MB）用首/中/尾采样 + 文件大小的组合 hash。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..config import IndexingConfig

MEDIA_TYPE_MAP = {
    # images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image",
    ".heic": "image", ".bmp": "image", ".tiff": "image",
    # videos
    ".mp4": "video", ".mov": "video", ".mkv": "video", ".avi": "video", ".webm": "video",
    # audio
    ".mp3": "audio", ".flac": "audio", ".wav": "audio",
    ".m4a": "audio", ".ogg": "audio", ".aac": "audio",
    # text
    ".txt": "text", ".md": "text", ".markdown": "text",
}


@dataclass
class FileInfo:
    path: str
    media_type: str  # image | video | audio
    file_hash: str
    mtime: float
    size: int


class FileScanner:
    def __init__(self, config: IndexingConfig):
        self.cfg = config
        self.state_db_path = config.state_db_path
        self.large_threshold = config.large_file_threshold_mb * 1024 * 1024
        self.hash_algorithm = config.hash_algorithm
        Path(self.state_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_state_db()

    # ------------------------------------------------------------------ #
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.state_db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_state_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path  TEXT PRIMARY KEY,
                    file_hash  TEXT NOT NULL,
                    mtime      REAL NOT NULL,
                    vector_ids TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_hash ON indexed_files(file_hash)"
            )

    # ------------------------------------------------------------------ #
    def scan_directory(self, root_dir: str, recursive: bool = True) -> list[FileInfo]:
        root = Path(root_dir)
        if not root.exists():
            logger.warning("Data source not found, skip: {}", root_dir)
            return []

        files: list[FileInfo] = []
        iterator = root.rglob("*") if recursive else root.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            media_type = MEDIA_TYPE_MAP.get(path.suffix.lower())
            if media_type is None:
                continue
            try:
                stat = path.stat()
            except OSError as e:
                logger.warning("Cannot stat {}: {}", path, e)
                continue
            files.append(
                FileInfo(
                    path=str(path),
                    media_type=media_type,
                    file_hash="",  # 按需计算
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
            )
        logger.info("Scanned {}: {} media files", root_dir, len(files))
        return files

    # ------------------------------------------------------------------ #
    def detect_changes(
        self, files: list[FileInfo]
    ) -> tuple[list[FileInfo], list[FileInfo], list[str]]:
        """返回 (new_files, modified_files, deleted_paths)。"""
        with self._conn() as conn:
            indexed = {
                row[0]: {"mtime": row[1], "file_hash": row[2]}
                for row in conn.execute(
                    "SELECT file_path, mtime, file_hash FROM indexed_files"
                )
            }

        current_paths = set()
        new_files: list[FileInfo] = []
        modified_files: list[FileInfo] = []

        for f in files:
            current_paths.add(f.path)
            if f.path not in indexed:
                f.file_hash = self.compute_hash(f.path)
                new_files.append(f)
            elif f.mtime != indexed[f.path]["mtime"]:
                # mtime 变了，重新 hash 确认内容是否真变
                f.file_hash = self.compute_hash(f.path)
                if f.file_hash != indexed[f.path]["file_hash"]:
                    modified_files.append(f)

        deleted_paths = [p for p in indexed if p not in current_paths]

        logger.info(
            "Changes: {} new, {} modified, {} deleted",
            len(new_files), len(modified_files), len(deleted_paths),
        )
        return new_files, modified_files, deleted_paths

    # ------------------------------------------------------------------ #
    def compute_hash(self, file_path: str, chunk_size: int = 1 << 16) -> str:
        """计算文件哈希；大文件用采样 hash。"""
        file_size = os.path.getsize(file_path)
        h = self._new_hasher()

        if file_size <= self.large_threshold:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
        else:
            sample = 1 << 20  # 1MB
            with open(file_path, "rb") as f:
                h.update(f.read(sample))               # 首部
                f.seek(file_size // 2)
                h.update(f.read(sample))               # 中间
                f.seek(max(0, file_size - sample))
                h.update(f.read(sample))               # 尾部
            h.update(str(file_size).encode())          # 文件大小作为额外熵
        return h.hexdigest()

    def _new_hasher(self):
        if self.hash_algorithm == "xxhash":
            import xxhash

            return xxhash.xxh64()
        if self.hash_algorithm == "md5":
            return hashlib.md5()
        return hashlib.sha256()

    # ------------------------------------------------------------------ #
    def mark_indexed(self, file_info: FileInfo, vector_ids: list[str]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO indexed_files
                   (file_path, file_hash, mtime, vector_ids, indexed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    file_info.path,
                    file_info.file_hash,
                    file_info.mtime,
                    ",".join(vector_ids),
                    time.time(),
                ),
            )

    def get_vector_ids(self, file_path: str) -> list[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vector_ids FROM indexed_files WHERE file_path = ?", (file_path,)
            ).fetchone()
        if row and row[0]:
            return row[0].split(",")
        return []

    def remove_indexed(self, file_path: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path,))
