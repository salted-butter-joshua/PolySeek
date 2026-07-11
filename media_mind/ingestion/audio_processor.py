"""音频转写处理器（Whisper）。

为什么音频走"转写 → 文本 Embedding"而不是直接音频 Embedding：
1. CLIP 系列没有原生音频编码器；引 CLAP 会增加一套模型。
2. Whisper 中文转写质量高（WER < 10%），转成文本后进入 CLIP 文本编码器，
   与图片在同一语义空间，天然支持"文搜音频""图搜音频"。
3. 文本转写还能做全文检索 fallback。

分段：Whisper 原生按 silence + 30s 窗口切分，每段独立生成一条文本 Embedding，
metadata 带 start/end，检索时定位到音频片段。
"""

from __future__ import annotations

from loguru import logger

from ..config import AudioConfig


class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.cfg = config
        self.model_name = config.whisper_model
        self.language = config.language
        self.min_chars = config.min_segment_chars
        self._model = None  # 懒加载

    @property
    def model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "缺少 openai-whisper，请安装：pip install 'media-mind[audio]'"
                ) from e
            logger.info("Loading Whisper model: {} ...", self.model_name)
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded.")
        return self._model

    def transcribe(self, audio_path: str) -> list[dict]:
        """转写音频，返回 [{'text','start','end'}, ...]。"""
        try:
            result = self.model.transcribe(
                audio_path, language=self.language, verbose=False
            )
        except Exception as e:
            logger.error("Whisper transcription failed for {}: {}", audio_path, e)
            return []

        segments: list[dict] = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if len(text) < self.min_chars:
                continue
            segments.append({"text": text, "start": seg["start"], "end": seg["end"]})

        logger.debug("Transcribed {}: {} segments", audio_path, len(segments))
        return segments
