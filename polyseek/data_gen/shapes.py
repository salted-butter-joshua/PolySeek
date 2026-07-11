"""用 PIL 绘制概念图片：在背景上画一个大号彩色形状，叠加轻噪声增加真实感与文件体积。"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int = 5) -> list:
    pts = []
    for i in range(n * 2):
        ang = math.pi / n * i - math.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _heart_points(cx: float, cy: float, s: float) -> list:
    pts = []
    for deg in range(0, 360, 6):
        t = math.radians(deg)
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * s / 16.0, cy - y * s / 16.0))
    return pts


def draw_concept_image(
    color_rgb: tuple[int, int, int],
    shape_key: str,
    size: int = 512,
    seed: int = 0,
    noise: float = 12.0,
) -> Image.Image:
    """绘制一张概念图片（确定性依赖 seed）。"""
    rng = random.Random(seed)

    # 中性背景（浅/深随机），与彩色形状形成对比
    bg = rng.choice([(245, 245, 245), (225, 228, 232), (35, 38, 42), (60, 62, 66)])
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    # 形状大小与位置随机（保持大号，让颜色成为主导信号）
    r = rng.uniform(0.30, 0.42) * size
    cx = rng.uniform(r, size - r)
    cy = rng.uniform(r, size - r)
    fill = color_rgb

    if shape_key == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape_key == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape_key == "triangle":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill
        )
    elif shape_key == "diamond":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy), (cx, cy + r), (cx + r, cy)], fill=fill
        )
    elif shape_key == "star":
        draw.polygon(_star_points(cx, cy, r, r * 0.45), fill=fill)
    elif shape_key == "heart":
        draw.polygon(_heart_points(cx, cy, r), fill=fill)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    # 叠加轻噪声：提升真实感，同时让 JPEG 体积更接近真实照片（便于用体积撑数据量）
    if noise > 0:
        arr = np.asarray(img).astype(np.int16)
        n = np.random.default_rng(seed).normal(0, noise, arr.shape)
        arr = np.clip(arr + n, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img
