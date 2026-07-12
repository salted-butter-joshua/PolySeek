#!/usr/bin/env python
"""从 HuggingFace（走 HF_ENDPOINT 镜像）下载 Flickr30k 原始图片并按 <filename> 保存。

数据集 nlphuji/flickr30k：全量 31783 张图都在 "test" split，字段含 image / filename。
支持断点续传（已存在的文件跳过）。

用法（容器内，datasets 未装会自动提示）：
    pip install -q datasets
    python scripts/download_flickr30k_images.py --out /raw
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 Flickr30k 原图")
    ap.add_argument("--out", required=True, help="输出目录（保存为 <filename>.jpg）")
    ap.add_argument("--dataset", default="nlphuji/flickr30k")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("缺少 datasets：先执行 pip install datasets")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)
    saved = skipped = 0
    for i, ex in enumerate(ds):
        fname = ex["filename"]
        path = out / fname
        if path.exists():
            skipped += 1
            continue
        ex["image"].convert("RGB").save(path)
        saved += 1
        if i % 2000 == 0:
            print(f"{i}/{total} (saved={saved}, skipped={skipped})", flush=True)

    print(f"done: total={total}, saved={saved}, skipped={skipped} -> {out}")


if __name__ == "__main__":
    main()
