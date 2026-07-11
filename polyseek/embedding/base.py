"""Embedding 服务抽象基类。

核心设计原则：所有后端（Chinese-CLIP / SigLIP 2 / OpenAI CLIP）都实现同一接口，
上层代码完全不感知具体模型，换模型只需改配置。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

ImageInput = str | Path | Image.Image


class EmbeddingService(ABC):
    """Embedding 服务抽象基类。

    约定：
    - 所有 ``encode_*`` 方法返回 **L2 归一化** 后的 float32 向量，
      使得 Cosine Similarity 等价于点积，便于向量库统一用 IP/COSINE 度量。
    - 单条编码返回 ``[d]``，批量编码返回 ``[N, d]``。
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.dimension: int = 0  # 子类初始化时设置

    # ------------------------------------------------------------------ #
    # 抽象接口
    # ------------------------------------------------------------------ #
    @abstractmethod
    def encode_image(self, image: ImageInput) -> np.ndarray:
        """单张图片 → 归一化向量 ``[d]``。接受文件路径或 PIL Image。"""
        ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """单条文本 → 归一化向量 ``[d]``。"""
        ...

    @abstractmethod
    def encode_images_batch(
        self, images: list[ImageInput]
    ) -> tuple[np.ndarray, list[int]]:
        """批量图片 → ``(vectors [M, d], kept_indices)``。

        Ingestion 的性能关键路径。为避免坏图污染检索结果，读取失败的图片会被
        **跳过**（而非填零向量）。因此返回值同时给出：

        - ``vectors``: 形状 ``[M, d]``（``M <= len(images)``）的归一化矩阵；
        - ``kept_indices``: 长度 ``M`` 的列表，``vectors[i]`` 对应
          ``images[kept_indices[i]]``。调用方据此对齐 metadata，避免错位。
        """
        ...

    @abstractmethod
    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        """批量文本 → 归一化向量矩阵 ``[N, d]``。"""
        ...

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化，使 Cosine Similarity = Dot Product。数值稳定，防除零。"""
        vectors = vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms

    @staticmethod
    def _load_pil(image: ImageInput) -> Image.Image:
        """把路径或 PIL Image 统一成 RGB PIL Image。"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"{self.__class__.__name__}(model_name={self.model_name!r}, "
            f"device={self.device!r}, dimension={self.dimension})"
        )
