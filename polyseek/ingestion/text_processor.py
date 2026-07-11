"""纯文本文档处理器（用于文搜文）。

把文档切成短块后，用 CLIP/SigLIP 的文本编码器编码进同一语义空间。为什么要切块：
CLIP 文本编码器有 token 上限（Chinese-CLIP ~52，SigLIP/OpenAI ~64/77），
整篇长文直接编码会截断丢信息。按字符窗口 + 重叠切块，检索时可定位到具体段落。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..config import TextConfig


class TextProcessor:
    def __init__(self, config: TextConfig):
        self.cfg = config
        self.chunk_chars = config.chunk_chars
        self.overlap = min(config.chunk_overlap, config.chunk_chars - 1)
        self.min_chars = config.min_chunk_chars
        self.encoding = config.encoding

    def chunk_file(self, path: str) -> list[dict]:
        """读取文件并切块，返回 [{'text','start_char','end_char'}, ...]。"""
        try:
            content = Path(path).read_text(encoding=self.encoding, errors="ignore")
        except Exception as e:
            logger.error("Cannot read text file {}: {}", path, e)
            return []

        content = content.strip()
        if len(content) < self.min_chars:
            return []

        chunks: list[dict] = []
        step = max(1, self.chunk_chars - self.overlap)
        for start in range(0, len(content), step):
            piece = content[start : start + self.chunk_chars].strip()
            if len(piece) < self.min_chars:
                continue
            chunks.append(
                {
                    "text": piece,
                    "start_char": start,
                    "end_char": min(start + self.chunk_chars, len(content)),
                }
            )
            if start + self.chunk_chars >= len(content):
                break

        logger.debug("Chunked {}: {} chunks", path, len(chunks))
        return chunks
