#!/usr/bin/env python
"""把"图片→中文描述"的公开数据集（如 Flickr30k-CN / COCO-CN）转成 PolySeek 评测集。

输入：JSON 数组或 JSONL，每条含一个图片文件名 + 一条或多条中文 caption。
输出：dataset.json，每条 caption 生成一个 text2image 用例，query=caption，
      relevant_files=[该图]（run_eval 按 basename 精确判命中）。

先把你的数据集整理成如下简单格式（字段名可用参数指定）：
    [{"image": "1000092795.jpg", "captions": ["两个年轻人在草地上", "..."]}, ...]

用法：
    python scripts/build_eval_from_captions.py --in flickr30k_cn.json \\
        --image-field image --caption-field captions --out eval/dataset.json --sample 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="公开图文数据集 → PolySeek 评测集")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="eval/dataset.json")
    ap.add_argument("--image-field", default="image")
    ap.add_argument("--caption-field", default="captions",
                    help="值可为字符串或字符串列表")
    ap.add_argument("--sample", type=int, default=0, help=">0 时随机抽样这么多条 caption")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = _load(args.inp)
    cases: list[dict] = []
    for it in items:
        image = it.get(args.image_field)
        if not image:
            continue
        caps = it.get(args.caption_field, [])
        if isinstance(caps, str):
            caps = [caps]
        for cap in caps:
            cap = str(cap).strip()
            if not cap:
                continue
            cases.append({
                "mode": "text2image",
                "query": cap,
                "target_media_type": "image",
                "relevant_files": [Path(image).name],
            })

    if args.sample and args.sample < len(cases):
        random.Random(args.seed).shuffle(cases)
        cases = cases[: args.sample]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(cases)} 条 text2image 用例 -> {out}")


if __name__ == "__main__":
    main()
