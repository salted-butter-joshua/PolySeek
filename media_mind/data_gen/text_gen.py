"""为每个概念生成中文文本文档（用于文搜文）。"""

from __future__ import annotations

import random

from .concepts import Concept

_TEMPLATES = [
    "这是一段关于{phrase}的描述。画面中最醒目的是{color}，形状呈现为{shape}。",
    "在这张图里，{shape}占据了主要位置，整体色调偏{color}，给人鲜明的视觉印象。",
    "{color}是这幅作品的主色。作者用一个{shape}来表达简洁而有力的构图。",
    "如果要检索{phrase}，关键词包括{color}、{shape}以及它们的组合。",
    "摄影笔记：主体是一个{shape}，颜色为{color}，背景做了中性化处理以突出主体。",
    "设计说明：本图采用{color}作为强调色，主体几何形状为{shape}，风格极简。",
    "孩子指着屏幕说，这是一个{color}的{shape}，我最喜欢这种{color}了。",
    "在色彩心理学里，{color}常与特定情绪相关；配合{shape}的轮廓，观感更加统一。",
]

_EXTRA = [
    "它可以出现在海报、图标或插画中。",
    "这种组合在数据集里被反复使用，用于验证检索系统的召回能力。",
    "无论放大还是缩小，主体的颜色和形状都清晰可辨。",
    "这段文字本身也会被编码进向量空间，用于文搜文实验。",
    "相似的文档应当在检索时聚集在一起。",
]


def generate_document(concept: Concept, seed: int, min_chars: int = 120) -> str:
    rng = random.Random(seed)
    lines: list[str] = []
    while sum(len(x) for x in lines) < min_chars:
        tpl = rng.choice(_TEMPLATES)
        line = tpl.format(
            phrase=concept.phrase, color=concept.color_name, shape=concept.shape_name
        )
        line += rng.choice(_EXTRA)
        lines.append(line)
    title = f"# {concept.phrase}\n\n"
    return title + "\n\n".join(lines) + "\n"
