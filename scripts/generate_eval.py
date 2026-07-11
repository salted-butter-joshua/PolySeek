#!/usr/bin/env python
"""从数据清单生成 100 条评测用例，覆盖文搜文 / 文搜图 / 图搜图。

用例判定命中的方式：解析结果文件名的概念（``{颜色}_{形状}``），与用例目标概念比较，
因此与索引时的绝对路径无关，跨环境稳定。

用法：
    python scripts/generate_eval.py --manifest sample_data/manifest.json --out eval/dataset.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def concept_of(path: str) -> str:
    parts = Path(path).name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="生成评测用例")
    ap.add_argument("--manifest", default="sample_data/manifest.json")
    ap.add_argument("--out", default="eval/dataset.json")
    ap.add_argument("--n", type=int, default=100, help="用例总数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    files = manifest["files"]

    by_mod_concept: dict[str, dict[str, list[str]]] = {
        "text": defaultdict(list), "image": defaultdict(list),
        "video": defaultdict(list), "audio": defaultdict(list),
    }
    for f in files:
        by_mod_concept.setdefault(f["modality"], defaultdict(list))[f["concept"]].append(f["path"])

    rng = random.Random(args.seed)
    cases: list[dict] = []

    # 概念 -> (颜色, 形状) 短语，用于构造文本 query
    def phrase(concept: str) -> str:
        color, shape = concept.split("_", 1)
        return f"{color}的{shape}"

    text_concepts = [c for c, v in by_mod_concept["text"].items() if v]
    image_concepts = [c for c, v in by_mod_concept["image"].items() if v]

    n_t2t = args.n // 3
    n_t2i = args.n // 3
    n_i2i = args.n - n_t2t - n_t2i

    # 文搜文：query=概念短语，relevant=该概念的所有文本文档
    for _ in range(n_t2t):
        if not text_concepts:
            break
        c = rng.choice(text_concepts)
        cases.append({
            "mode": "text2text", "query": phrase(c),
            "target_media_type": "text", "target_concept": c,
        })

    # 文搜图：query=概念短语，relevant=该概念的图片（颜色对齐提供信号）
    for _ in range(n_t2i):
        if not image_concepts:
            break
        c = rng.choice(image_concepts)
        cases.append({
            "mode": "text2image", "query": phrase(c),
            "target_media_type": "image", "target_concept": c,
        })

    # 图搜图：query=某概念的一张图，relevant=同概念其它图
    for _ in range(n_i2i):
        if not image_concepts:
            break
        c = rng.choice(image_concepts)
        imgs = by_mod_concept["image"][c]
        if len(imgs) < 2:
            continue
        q = rng.choice(imgs)
        cases.append({
            "mode": "image2image", "query_image": q,
            "target_media_type": "image", "target_concept": c,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    by_mode: dict[str, int] = defaultdict(int)
    for c in cases:
        by_mode[c["mode"]] += 1
    print(f"生成 {len(cases)} 条用例 {dict(by_mode)} -> {out}")


if __name__ == "__main__":
    main()
