"""数据摄取管道：编排 Scanner → Processor → Embedding → VectorStore。

数据流：
- 文本: path → chunk → 批量 Text Encoder → 多条 vector(带 char 偏移) → insert
- 图片: path → Image Encoder → vector → insert
- 视频: path → extract_frames → 批量 Image Encoder → 多条 vector(带 frame_ts) → insert
- 音频: path → Whisper transcribe → 批量 Text Encoder → 多条 vector(带 start/end) → insert

同时记录每种类型的预处理/嵌入耗时（IngestionStats），供前端展示离线嵌入总时间。
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..config import AppConfig, DataSource
from ..embedding.base import EmbeddingService
from ..storage.base import VectorStore
from .audio_processor import AudioProcessor
from .image_processor import ImageProcessor
from .scanner import FileInfo, FileScanner
from .stats import IngestionStats
from .text_processor import TextProcessor
from .video_processor import VideoProcessor


class IngestionPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        scanner: FileScanner,
        config: AppConfig,
    ):
        self.embed = embedding_service
        self.store = vector_store
        self.scanner = scanner
        self.config = config
        self.batch_size = config.embedding.batch_size
        self.image_processor = ImageProcessor(config.image)
        self.video_processor = VideoProcessor(config.video)
        self.audio_processor = AudioProcessor(config.audio)
        self.text_processor = TextProcessor(config.text)
        self.stats_path = config.indexing.stats_path
        self.stats = IngestionStats.load(self.stats_path)

    # ------------------------------------------------------------------ #
    def _collect_files(self, data_sources: list[DataSource]) -> list[FileInfo]:
        all_files: list[FileInfo] = []
        for source in data_sources:
            files = self.scanner.scan_directory(source.path, recursive=source.recursive)
            allowed = set(source.media_types)
            all_files.extend(f for f in files if f.media_type in allowed)
        return all_files

    def run_full_index(self, data_sources: list[DataSource]) -> None:
        wall0 = time.perf_counter()
        all_files = self._collect_files(data_sources)
        logger.info("Total files to index: {}", len(all_files))

        for f in tqdm(all_files, desc="Hashing"):
            if not f.file_hash:
                f.file_hash = self.scanner.compute_hash(f.path)

        self._dispatch(all_files)
        self._finalize(wall0)
        logger.info("Full indexing complete. Total vectors: {}", self.store.count())

    def run_incremental_index(self, data_sources: list[DataSource]) -> None:
        wall0 = time.perf_counter()
        all_files = self._collect_files(data_sources)
        new_files, modified_files, deleted_paths = self.scanner.detect_changes(all_files)

        for path in deleted_paths:
            self._remove_file(path)
        for f in modified_files:
            self._remove_file(f.path)

        to_index = new_files + modified_files
        self._dispatch(to_index)
        self._finalize(wall0)

        logger.info(
            "Incremental done. Added files: {}, Deleted: {}, Total vectors: {}",
            len(to_index), len(deleted_paths), self.store.count(),
        )

    def _finalize(self, wall0: float) -> None:
        self.stats.total_wall_seconds += time.perf_counter() - wall0
        self.stats.save(self.stats_path)

    def _remove_file(self, path: str) -> None:
        vector_ids = self.scanner.get_vector_ids(path)
        if vector_ids:
            self.store.delete(vector_ids)
        self.scanner.remove_indexed(path)

    def _dispatch(self, files: list[FileInfo]) -> None:
        buckets: dict[str, list[FileInfo]] = {
            "text": [], "image": [], "video": [], "audio": []
        }
        for f in files:
            buckets.setdefault(f.media_type, []).append(f)
        if buckets["text"]:
            self._index_texts(buckets["text"])
        if buckets["image"]:
            self._index_images(buckets["image"])
        if buckets["video"]:
            self._index_videos(buckets["video"])
        if buckets["audio"]:
            self._index_audios(buckets["audio"])

    # ------------------------------------------------------------------ #
    def _index_texts(self, files: list[FileInfo]) -> None:
        logger.info("Indexing {} text documents ...", len(files))
        for f in tqdm(files, desc="Text"):
            with self.stats.timer_process("text"):
                chunks = self.text_processor.chunk_file(f.path)
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            with self.stats.timer_embed("text", vectors=len(texts)):
                vectors = self.embed.encode_texts_batch(texts)

            metadatas = [
                {
                    "media_type": "text",
                    "file_path": f.path,
                    "file_hash": f.file_hash,
                    "mtime": f.mtime,
                    "chunk_start": c["start_char"],
                    "chunk_end": c["end_char"],
                    "chunk_text": c["text"],
                }
                for c in chunks
            ]
            ids = self.store.insert(vectors, metadatas)
            self.scanner.mark_indexed(f, ids)
            self.stats.add_files("text")

    def _index_images(self, files: list[FileInfo]) -> None:
        logger.info("Indexing {} images ...", len(files))
        for start in tqdm(range(0, len(files), self.batch_size), desc="Images"):
            batch = files[start : start + self.batch_size]
            loaded: list = []
            valid_batch: list[FileInfo] = []
            with self.stats.timer_process("image"):
                for f in batch:
                    try:
                        loaded.append(self.image_processor.load(f.path))
                        valid_batch.append(f)
                    except Exception as e:
                        logger.warning("Skip unreadable image {}: {}", f.path, e)

            if not loaded:
                continue

            with self.stats.timer_embed("image"):
                vectors, kept = self.embed.encode_images_batch(loaded)
            if len(vectors) == 0:
                continue

            kept_files = [valid_batch[i] for i in kept]
            metadatas = [
                {
                    "media_type": "image",
                    "file_path": f.path,
                    "file_hash": f.file_hash,
                    "mtime": f.mtime,
                }
                for f in kept_files
            ]
            ids = self.store.insert(vectors, metadatas)
            for f, vid in zip(kept_files, ids, strict=True):
                self.scanner.mark_indexed(f, [vid])
            self.stats.per_type["image"].vectors += len(ids)
            self.stats.add_files("image", len(kept_files))

    def _index_videos(self, files: list[FileInfo]) -> None:
        logger.info("Indexing {} videos ...", len(files))
        for f in tqdm(files, desc="Videos"):
            with self.stats.timer_process("video"):
                frames = self.video_processor.extract_frames(f.path)
            if not frames:
                continue

            all_vectors: list[np.ndarray] = []
            all_metadatas: list[dict] = []
            for start in range(0, len(frames), self.batch_size):
                chunk = frames[start : start + self.batch_size]
                imgs = [img for img, _ in chunk]
                with self.stats.timer_embed("video"):
                    vectors, kept = self.embed.encode_images_batch(imgs)
                if len(vectors) == 0:
                    continue
                for row, idx in enumerate(kept):
                    _, ts = chunk[idx]
                    all_vectors.append(vectors[row])
                    all_metadatas.append(
                        {
                            "media_type": "video_frame",
                            "file_path": f.path,
                            "file_hash": f.file_hash,
                            "mtime": f.mtime,
                            "frame_ts": ts,
                        }
                    )

            if all_vectors:
                ids = self.store.insert(np.stack(all_vectors), all_metadatas)
                self.scanner.mark_indexed(f, ids)
                self.stats.per_type["video"].vectors += len(ids)
                self.stats.add_files("video")

    def _index_audios(self, files: list[FileInfo]) -> None:
        logger.info("Indexing {} audio files ...", len(files))
        for f in tqdm(files, desc="Audio"):
            with self.stats.timer_process("audio"):
                segments = self.audio_processor.transcribe(f.path)
            if not segments:
                continue

            texts = [seg["text"] for seg in segments]
            with self.stats.timer_embed("audio", vectors=len(texts)):
                vectors = self.embed.encode_texts_batch(texts)
            metadatas = [
                {
                    "media_type": "audio_transcript",
                    "file_path": f.path,
                    "file_hash": f.file_hash,
                    "mtime": f.mtime,
                    "segment_start": seg["start"],
                    "segment_end": seg["end"],
                    "transcript_text": seg["text"],
                }
                for seg in segments
            ]
            ids = self.store.insert(vectors, metadatas)
            self.scanner.mark_indexed(f, ids)
            self.stats.add_files("audio")
