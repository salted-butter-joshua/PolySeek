"""OpenAI CLIP 后端实现（基于 HuggingFace transformers）。

文档工厂函数引用了该后端，这里补全。主要用于英文场景或与原版 CLIP 做对比基线。

常用 checkpoint：
- openai/clip-vit-base-patch16:  d=512
- openai/clip-vit-large-patch14: d=768
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from .base import EmbeddingService, ImageInput


class OpenAIClipEmbedding(EmbeddingService):
    """基于 transformers 的 OpenAI CLIP 实现。"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: str = "cpu",
        cache_dir: str = "./models",
    ):
        super().__init__(model_name, device)
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "缺少 transformers，请安装：pip install 'media-mind[siglip]'"
            ) from e

        logger.info("Loading OpenAI CLIP {} on {} ...", model_name, device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.model.eval()

        self.dimension = int(self.model.config.projection_dim)
        logger.info("OpenAI CLIP loaded. dimension={}", self.dimension)

    @torch.no_grad()
    def encode_image(self, image: ImageInput) -> np.ndarray:
        img = self._load_pil(image)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs).cpu().numpy()
        return self.normalize(features).squeeze(0)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        features = self.model.get_text_features(**inputs).cpu().numpy()
        return self.normalize(features).squeeze(0)

    @torch.no_grad()
    def encode_images_batch(
        self, images: list[ImageInput]
    ) -> tuple[np.ndarray, list[int]]:
        pil_images = []
        kept: list[int] = []
        for i, img in enumerate(images):
            try:
                pil_images.append(self._load_pil(img))
                kept.append(i)
            except Exception as e:
                logger.warning("Skip unreadable image {}: {}", img, e)

        if not pil_images:
            return np.empty((0, self.dimension), dtype=np.float32), []

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs).cpu().numpy()
        return self.normalize(features), kept

    @torch.no_grad()
    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs).cpu().numpy()
        return self.normalize(features)
