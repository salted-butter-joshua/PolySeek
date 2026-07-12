#!/usr/bin/env python
"""从 HuggingFace（走 HF_ENDPOINT 镜像）下载 Flickr30k 原始图片并按 <filename> 保存。

数据集 nlphuji/flickr30k：全量 31783 张图都在 "test" split，字段含 image / filename。
支持断点续传（已存在的文件跳过）。

注意：nlphuji/flickr30k 是“脚本型”数据集，datasets 3.x 已不支持，
必须安装 2.x 并信任远程脚本：
    pip install -q "datasets==2.21.0"
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
        import datasets as _ds_lib
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit('缺少 datasets：先执行 pip install "datasets==2.21.0"') from e

    if int(_ds_lib.__version__.split(".")[0]) >= 3:
        raise SystemExit(
            f"datasets {_ds_lib.__version__} 不支持脚本型数据集（{args.dataset}）。"
            '请降级：pip install "datasets==2.21.0"'
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
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
