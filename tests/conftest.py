"""测试夹具：用轻量 Fake 实现替代重模型/外部 DB，使单测无需下载权重或起服务。"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pytest

from polyseek.embedding.base import EmbeddingService, ImageInput
from polyseek.storage.base import SearchFilter, SearchResult, VectorStore


class FakeEmbedding(EmbeddingService):
    """确定性伪 Embedding：对输入做 hash → 稳定伪随机向量并归一化。

    相同输入永远得到相同向量，便于断言检索命中；无需任何深度学习依赖。
    """

    def __init__(self, dimension: int = 32):
        super().__init__(model_name="fake", device="cpu")
        self.dimension = dimension

    def _vec(self, key: str) -> np.ndarray:
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return self.normalize(rng.standard_normal(self.dimension).astype(np.float32))

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self._vec(f"img::{image}")

    def encode_text(self, text: str) -> np.ndarray:
        return self._vec(f"txt::{text}")

    def encode_images_batch(self, images: list[ImageInput]):
        vecs = np.stack([self._vec(f"img::{i}") for i in images]) if images else \
            np.empty((0, self.dimension), np.float32)
        return vecs, list(range(len(images)))

    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), np.float32)
        return np.stack([self._vec(f"txt::{t}") for t in texts])


class InMemoryStore(VectorStore):
    """内存向量库：暴力 cosine 检索，实现 VectorStore 全部接口，供测试用。"""

    def __init__(self):
        self._vecs: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}

    def create_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def insert(self, vectors: np.ndarray, metadatas: list[dict]) -> list[str]:
        ids = []
        for v, m in zip(vectors, metadatas, strict=False):
            i = str(uuid.uuid4())
            self._vecs[i] = np.asarray(v, dtype=np.float32)
            self._meta[i] = dict(m)
            ids.append(i)
        return ids

    def search(self, query_vector, top_k=20, filters: SearchFilter | None = None):
        q = np.asarray(query_vector, dtype=np.float32)
        scored = []
        for i, v in self._vecs.items():
            m = self._meta[i]
            if filters and not self._match(m, filters):
                continue
            scored.append((i, float(np.dot(q, v))))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i, s in scored[:top_k]:
            m = self._meta[i]
            out.append(
                SearchResult(
                    id=i, score=s,
                    file_path=m.get("file_path", ""),
                    media_type=m.get("media_type", ""),
                    metadata={k: v for k, v in m.items()
                              if k not in ("file_path", "media_type")},
                )
            )
        return out

    @staticmethod
    def _match(m: dict, f: SearchFilter) -> bool:
        if f.media_type and m.get("media_type") != f.media_type:
            return False
        if f.mtime_min is not None and m.get("mtime", 0) < f.mtime_min:
            return False
        if f.mtime_max is not None and m.get("mtime", 0) > f.mtime_max:
            return False
        return True

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._vecs.pop(i, None)
            self._meta.pop(i, None)

    def count(self) -> int:
        return len(self._vecs)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_embedding():
    return FakeEmbedding(dimension=32)


@pytest.fixture
def memory_store():
    store = InMemoryStore()
    store.create_collection(32)
    return store
