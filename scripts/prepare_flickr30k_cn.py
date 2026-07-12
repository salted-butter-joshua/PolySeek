#!/usr/bin/env python
"""准备 Flickr30k-CN 数据集：解出图片 + 直接生成 PolySeek 评测集。

输入是 Chinese-CLIP 官方发布的检索格式（OFA-Sys/Chinese-CLIP 的 datasets）：
    {split}_imgs.tsv    每行： image_id \\t base64编码的图片
    {split}_texts.jsonl 每行： {"text_id": int, "text": "中文描述", "image_ids": [int, ...]}

`image_ids` 就是该 caption 的 ground-truth 图片，天然对应文搜图的正确答案。

用法：
    python scripts/prepare_flickr30k_cn.py \\
        --data-dir /data/flickr30k-cn --split test \\
        --out-images /data/flickr30k-cn/images --extract-images \\
        --out-eval eval/flickr30k_cn.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def extract_images(imgs_tsv: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(imgs_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            iid, b64 = line.split("\t", 1)
            (out_dir / f"{iid}.jpg").write_bytes(base64.b64decode(b64))
            n += 1
            if n % 2000 == 0:
                print(f"  ...已解出 {n} 张")
    print(f"解出 {n} 张图 -> {out_dir}")
    return n


def build_eval(texts_jsonl: Path, out_eval: Path) -> int:
    cases = []
    with open(texts_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            iids = obj.get("image_ids", [])
            if not text or not iids:
                continue
            cases.append({
                "mode": "text2image",
                "query": text,
                "target_media_type": "image",
                "relevant_files": [f"{i}.jpg" for i in iids],
            })
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(cases)} 条 text2image 评测用例 -> {out_eval}")
    return len(cases)


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 Flickr30k-CN：解图 + 生成评测集")
    ap.add_argument("--data-dir", required=True, help="含 {split}_imgs.tsv 和 {split}_texts.jsonl 的目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-images", default=None, help="图片解出目录（用于索引）")
    ap.add_argument("--extract-images", action="store_true", help="从 tsv 解出图片（大文件，几分钟）")
    ap.add_argument("--out-eval", default="eval/flickr30k_cn.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    imgs_tsv = data_dir / f"{args.split}_imgs.tsv"
    texts_jsonl = data_dir / f"{args.split}_texts.jsonl"

    if args.extract_images:
        if not args.out_images:
            raise SystemExit("--extract-images 需同时指定 --out-images")
        extract_images(imgs_tsv, Path(args.out_images))

    build_eval(texts_jsonl, Path(args.out_eval))


if __name__ == "__main__":
    main()
