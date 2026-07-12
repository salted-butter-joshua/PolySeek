"""SigLIP 2 后端实现（Google, 2025.2，基于 HuggingFace transformers）。

相比 Chinese-CLIP 的优势：
1. Sigmoid Loss → 检索场景相似度评分更可靠、可跨 batch 比较
2. 109 种语言原生支持（含中文）
3. NaFlex 变体支持原生长宽比
4. zero-shot retrieval benchmark 上全面超越 CLIP

常用 checkpoint：
- google/siglip2-base-patch16-224:  适合 CPU
- google/siglip2-large-patch16-384: 建议 GPU
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from .base import EmbeddingService, ImageInput


def _as_tensor(out) -> torch.Tensor:
    """兼容不同 transformers 版本的 get_*_features 返回值。

    旧版直接返回张量；新版（如 SigLIP2 在新 transformers 里）返回
    BaseModelOutputWithPooling 之类的 ModelOutput，需取 pooler_output。
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    # 不做 last_hidden_state 均值池化兜底：那不是对齐空间的投影输出，
    # 静默用它会让图文空间错位、检索接近随机——宁可报错暴露问题。
    raise TypeError(f"无法从 {type(out).__name__} 提取特征张量（缺少 pooler_output）")


class SigLIPEmbedding(EmbeddingService):
    """基于 transformers 的 SigLIP / SigLIP 2 实现。"""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str = "cpu",
        cache_dir: str = "./models",
    ):
        super().__init__(model_name, device)
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "缺少 transformers，请安装：pip install 'polyseek[siglip]'"
            ) from e

        logger.info("Loading SigLIP model {} on {} ...", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.model.eval()

        # 用一次前向探测实际输出维度（不同 checkpoint 的投影维度不一）。
        with torch.no_grad():
            probe = self._tokenize(["probe"])
            feats = _as_tensor(self.model.get_text_features(**probe))
        self.dimension = int(feats.shape[-1])
        logger.info("SigLIP loaded. dimension={}", self.dimension)

    def _tokenize(self, texts: list[str]) -> dict:
        """SigLIP 文本必须 padding='max_length'（定长 64、无 attention mask 训练）。

        用动态 padding=True 会让文本嵌入偏离训练分布，图文空间错位、检索接近随机。
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode_image(self, image: ImageInput) -> np.ndarray:
        img = self._load_pil(image)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        features = _as_tensor(self.model.get_image_features(**inputs)).cpu().numpy()
        return self.normalize(features).squeeze(0)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self._tokenize([text])
        features = _as_tensor(self.model.get_text_features(**inputs)).cpu().numpy()
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
        features = _as_tensor(self.model.get_image_features(**inputs)).cpu().numpy()
        return self.normalize(features), kept

    @torch.no_grad()
    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self._tokenize(texts)
        features = _as_tensor(self.model.get_text_features(**inputs)).cpu().numpy()
        return self.normalize(features)
