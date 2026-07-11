"""Chinese-CLIP 后端实现。

为什么作为默认后端：
1. 中文语义理解远优于原版 CLIP（中文 benchmark 上 +15~20%）
2. 权重开源、社区活跃
3. API 与 OpenAI CLIP 高度一致，迁移成本低

模型规格：
- ViT-B-16:  d=512,  ~188M 参数, CPU ~50ms/image
- ViT-L-14:  d=768,  ~428M 参数, CPU ~150ms/image
- ViT-H-14:  d=1024, ~986M 参数, 建议 GPU
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from .base import EmbeddingService, ImageInput


class ChineseClipEmbedding(EmbeddingService):
    """基于 cn_clip 的 Chinese-CLIP 实现。"""

    MODEL_DIMENSIONS = {
        "ViT-B-16": 512,
        "ViT-L-14": 768,
        "ViT-L-14-336": 768,
        "ViT-H-14": 1024,
        "RN50": 1024,
    }

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        device: str = "cpu",
        cache_dir: str = "./models",
    ):
        super().__init__(model_name, device)
        try:
            import cn_clip.clip as clip
            from cn_clip.clip import load_from_name
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "缺少 cn-clip，请安装：pip install 'polyseek[chinese-clip]' "
                "或 pip install cn-clip"
            ) from e

        self._clip = clip
        logger.info("Loading Chinese-CLIP {} on {} ...", model_name, device)
        self.model, self.preprocess = load_from_name(
            model_name, device=device, download_root=cache_dir
        )
        self.model.eval()

        # 优先用模型实际输出维度，回退到已知映射表。
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 0)
        if self.dimension == 0:
            with torch.no_grad():
                probe = self.model.encode_text(clip.tokenize(["probe"]).to(device))
            self.dimension = int(probe.shape[-1])
        logger.info("Chinese-CLIP loaded. dimension={}", self.dimension)

    def _preprocess_image(self, image: ImageInput) -> torch.Tensor:
        img = self._load_pil(image)
        return self.preprocess(img).unsqueeze(0)

    @torch.no_grad()
    def encode_image(self, image: ImageInput) -> np.ndarray:
        tensor = self._preprocess_image(image).to(self.device)
        features = self.model.encode_image(tensor).cpu().numpy()
        return self.normalize(features).squeeze(0)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._clip.tokenize([text]).to(self.device)
        features = self.model.encode_text(tokens).cpu().numpy()
        return self.normalize(features).squeeze(0)

    @torch.no_grad()
    def encode_images_batch(
        self, images: list[ImageInput]
    ) -> tuple[np.ndarray, list[int]]:
        """批量图片编码，跳过坏图并返回保留下来的原始索引。"""
        tensors: list[torch.Tensor] = []
        kept: list[int] = []
        for i, img in enumerate(images):
            try:
                tensors.append(self._preprocess_image(img))
                kept.append(i)
            except Exception as e:
                logger.warning("Skip unreadable image {}: {}", img, e)

        if not tensors:
            return np.empty((0, self.dimension), dtype=np.float32), []

        batch = torch.cat(tensors, dim=0).to(self.device)
        features = self.model.encode_image(batch).cpu().numpy()
        return self.normalize(features), kept

    @torch.no_grad()
    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokens = self._clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(tokens).cpu().numpy()
        return self.normalize(features)
