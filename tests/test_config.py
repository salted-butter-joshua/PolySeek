"""配置加载与校验测试。"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from polyseek.config import AppConfig, load_config


def test_defaults():
    cfg = AppConfig()
    assert cfg.embedding.backend == "chinese_clip"
    assert cfg.vector_store.backend == "milvus_lite"
    assert cfg.embedding.batch_size == 32


def test_load_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent(
            """
            embedding:
              backend: siglip
              model_name: google/siglip2-base-patch16-224
              batch_size: 8
            vector_store:
              backend: qdrant
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.embedding.backend == "siglip"
    assert cfg.embedding.batch_size == 8
    assert cfg.vector_store.backend == "qdrant"


def test_env_override(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("embedding:\n  device: cpu\n", encoding="utf-8")
    monkeypatch.setenv("POLYSEEK__EMBEDDING__DEVICE", "cuda")
    cfg = load_config(p)
    assert cfg.embedding.device == "cuda"


def test_invalid_backend_rejected():
    with pytest.raises(ValidationError):
        AppConfig(embedding={"backend": "not_a_backend"})
