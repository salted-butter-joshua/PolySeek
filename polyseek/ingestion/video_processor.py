"""视频抽帧处理器（基于 ffmpeg / ffprobe）。

抽帧策略：固定间隔（每 N 秒 1 帧）。简单、帧数可预测，覆盖 90% "找特定画面"的需求。

为什么用 ffmpeg 而不是 OpenCV：
1. ffmpeg 支持几乎所有容器/编码，OpenCV 依赖编译期 codec。
2. seek 更准（keyframe-aware）。
3. 少引入 OpenCV 重依赖。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from loguru import logger
from PIL import Image

from ..config import VideoConfig


class VideoProcessor:
    def __init__(self, config: VideoConfig):
        self.cfg = config
        self.frame_interval = config.frame_interval_seconds
        self.max_frames = config.max_frames_per_video
        self.thumbnail_size = tuple(config.thumbnail_size)
        self.ffmpeg = config.ffmpeg_path
        self.ffprobe = config.ffprobe_path

    def get_duration(self, video_path: str) -> float | None:
        cmd = [
            self.ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except (subprocess.CalledProcessError, KeyError, ValueError, FileNotFoundError) as e:
            logger.error("ffprobe failed for {}: {}", video_path, e)
            return None

    def extract_frames(self, video_path: str) -> list[tuple[Image.Image, float]]:
        """抽取关键帧，返回 [(PIL.Image, timestamp_seconds), ...]。"""
        duration = self.get_duration(video_path)
        if duration is None or duration <= 0:
            return []

        timestamps: list[float] = []
        t = 0.0
        while t < duration and len(timestamps) < self.max_frames:
            timestamps.append(round(t, 3))
            t += self.frame_interval

        frames: list[tuple[Image.Image, float]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, ts in enumerate(timestamps):
                frame_path = Path(tmpdir) / f"frame_{i:06d}.jpg"
                cmd = [
                    self.ffmpeg, "-v", "quiet",
                    "-ss", str(ts), "-i", video_path,
                    "-vframes", "1", "-q:v", "2",
                    str(frame_path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, check=False)
                except FileNotFoundError:
                    logger.error("ffmpeg not found at '{}', abort video processing.", self.ffmpeg)
                    return frames

                if not frame_path.exists():
                    continue
                try:
                    img = Image.open(frame_path).convert("RGB")
                    img.thumbnail(self.thumbnail_size, Image.LANCZOS)
                    frames.append((img, ts))
                except Exception as e:
                    logger.warning("Failed to read frame @ {}s: {}", ts, e)

        logger.debug("Extracted {} frames from {}", len(frames), video_path)
        return frames
