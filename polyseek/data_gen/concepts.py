"""合成数据的概念定义：颜色 × 形状。

用"颜色"作为主要语义信号——CLIP/Chinese-CLIP 对颜色词与颜色图像的对齐很强，
因此在纯合成数据上也能得到真实的文搜图检索信号；"形状"提供第二维区分度，
让 image→image 聚类更干净。每个概念 = (颜色, 形状)，作为 ground-truth 标签。
"""

from __future__ import annotations

from dataclasses import dataclass

# 颜色：中文名 + RGB
COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("红色", (220, 40, 40)),
    ("橙色", (240, 140, 30)),
    ("黄色", (240, 210, 40)),
    ("绿色", (40, 180, 70)),
    ("青色", (40, 200, 200)),
    ("蓝色", (40, 90, 220)),
    ("紫色", (150, 50, 200)),
    ("粉色", (240, 120, 180)),
    ("棕色", (140, 90, 50)),
    ("黑色", (30, 30, 30)),
    ("白色", (235, 235, 235)),
    ("灰色", (130, 130, 130)),
]

# 形状：中文名 + 英文 key
SHAPES: list[tuple[str, str]] = [
    ("圆形", "circle"),
    ("方形", "square"),
    ("三角形", "triangle"),
    ("星形", "star"),
    ("菱形", "diamond"),
    ("心形", "heart"),
]


@dataclass(frozen=True)
class Concept:
    color_name: str
    color_rgb: tuple[int, int, int]
    shape_name: str
    shape_key: str

    @property
    def key(self) -> str:
        return f"{self.color_name}_{self.shape_name}"

    @property
    def phrase(self) -> str:
        return f"{self.color_name}的{self.shape_name}"


def all_concepts() -> list[Concept]:
    out: list[Concept] = []
    for cname, rgb in COLORS:
        for sname, skey in SHAPES:
            out.append(Concept(cname, rgb, sname, skey))
    return out
