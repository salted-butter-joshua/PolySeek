"""Embedding 契约测试（用 FakeEmbedding，不下载真实权重）。

真实后端的等价测试放在 tests/integration/（需要权重，默认不在 CI 跑）。
"""

from __future__ import annotations

import numpy as np


def test_encode_text_shape_and_norm(fake_embedding):
    vec = fake_embedding.encode_text("一只橙色的猫")
    assert vec.shape == (fake_embedding.dimension,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_encode_image_deterministic(fake_embedding):
    a = fake_embedding.encode_image("/x/cat.jpg")
    b = fake_embedding.encode_image("/x/cat.jpg")
    assert np.allclose(a, b)


def test_batch_returns_kept_indices(fake_embedding):
    vecs, kept = fake_embedding.encode_images_batch(["/a.jpg", "/b.jpg", "/c.jpg"])
    assert vecs.shape == (3, fake_embedding.dimension)
    assert kept == [0, 1, 2]


def test_empty_batch(fake_embedding):
    vecs, kept = fake_embedding.encode_images_batch([])
    assert vecs.shape == (0, fake_embedding.dimension)
    assert kept == []


def test_texts_batch_shape(fake_embedding):
    vecs = fake_embedding.encode_texts_batch(["a", "b"])
    assert vecs.shape == (2, fake_embedding.dimension)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
