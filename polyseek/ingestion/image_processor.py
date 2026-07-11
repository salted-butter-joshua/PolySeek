"""图片预处理工具。

CLIP/SigLIP 的 processor 内部已含 resize/normalize，这里只做 ingestion 前的
轻量校验与可选降采样（超大图先缩小，降低解码与内存开销）。
"""

from __future__ import annotations

from PIL import Image

from ..config import ImageConfig


class ImageProcessor:
    def __init__(self, config: ImageConfig):
        self.max_dimension = config.max_dimension
        self.supported_formats = {fmt.lower() for fmt in config.supported_formats}

    def load(self, path: str) -> Image.Image:
        """加载图片为 RGB，超大图按最长边降采样。可能抛异常，调用方需捕获。"""
        img = Image.open(path).convert("RGB")
        if max(img.size) > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.LANCZOS)
        return img
