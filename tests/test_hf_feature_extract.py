"""_as_tensor：兼容 transformers 新旧版本 get_*_features 返回值的提取逻辑。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from polyseek.embedding.siglip import _as_tensor  # noqa: E402


class _FakeOutput:
    def __init__(self, pooler_output=None, last_hidden_state=None):
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state


def test_tensor_passthrough():
    t = torch.randn(2, 8)
    assert _as_tensor(t) is t


def test_pooler_output_extracted():
    t = torch.randn(2, 8)
    out = _FakeOutput(pooler_output=t)
    assert _as_tensor(out) is t


def test_no_pooler_raises_instead_of_silent_meanpool():
    # 均值池化不是对齐空间的投影输出，静默兜底会让检索接近随机——必须报错
    out = _FakeOutput(last_hidden_state=torch.randn(2, 5, 8))
    with pytest.raises(TypeError):
        _as_tensor(out)


def test_unknown_type_raises():
    with pytest.raises(TypeError):
        _as_tensor(object())
