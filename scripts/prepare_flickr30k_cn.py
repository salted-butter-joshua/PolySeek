#!/usr/bin/env python
"""准备 Flickr30k-CN（li-xirong / VisualSearch 格式）：分词中文 caption → 评测集，
并从原始 Flickr30k 图片目录筛出该 split 的图片作为检索语料。

数据包每个 split 一个目录（如 flickr30kzhmbosontest），结构：
    ImageSets/<name>.txt              该 split 的图片 id 列表（每行一个 id）
    TextData/seg.<name>.caption.txt   分词中文描述，每行： <imgid>#<idx> 词 词 词 ...
    FeatureData/...                   预计算 ResNet 特征（本项目不用，自己算 CLIP）

注意：该包不含原始 JPG，需另外下载 Flickr30k 图片（文件名形如 <imgid>.jpg）。

用法：
    python scripts/prepare_flickr30k_cn.py \\
        --split-dir /data/flickr30k-cn/flickr30kzhmbosontest \\
        --images-src /data/flickr30k-images \\
        --out-images /data/flickr30k-cn/test-images \\
        --out-eval eval/flickr30k_cn.json
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path


def _find(split_dir: str, sub: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(str(Path(split_dir) / sub / pattern)))
    return hits[0] if hits else None


def _detokenize(cap: str) -> str:
    """还原分词 caption 为自然中文。

    该语料的 token 格式是 ``词:词性``（如 ``警车:n`` ``，:wd``），直接去空格会把
    词性标注留在文本里污染 query。逐 token 剥掉最后一个 ``:`` 后的词性再拼接；
    无词性的纯分词格式也兼容。
    """
    words = []
    for tok in cap.split():
        if ":" in tok:
            tok = tok.rsplit(":", 1)[0]
        if tok:
            words.append(tok)
    return "".join(words)


def parse_captions(cap_file: str) -> tuple[list[dict], set[str]]:
    cases: list[dict] = []
    imgids: set[str] = set()
    with open(cap_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # 左： <imgid>#<idx>   右： 分词 caption（空格分隔，token 可能带 :词性）
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            head, cap = parts
            imgid = head.split("#", 1)[0]
            text = _detokenize(cap)
            if not text:
                continue
            imgids.add(imgid)
            cases.append({
                "mode": "text2image", "query": text,
                "target_media_type": "image", "relevant_files": [f"{imgid}.jpg"],
            })
    return cases, imgids


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 Flickr30k-CN（VisualSearch 格式）")
    ap.add_argument("--split-dir", required=True, help="如 .../flickr30kzhmbosontest")
    ap.add_argument("--images-src", default=None, help="原始 Flickr30k 图片目录（<imgid>.jpg）")
    ap.add_argument("--out-images", default=None, help="筛出的语料图片目录（用于索引）")
    ap.add_argument("--out-eval", default="eval/flickr30k_cn.json")
    args = ap.parse_args()

    cap_file = _find(args.split_dir, "TextData", "seg.*.caption.txt")
    if not cap_file:
        raise SystemExit(f"未找到 TextData/seg.*.caption.txt in {args.split_dir}")
    cases, imgids = parse_captions(cap_file)
    print(f"解析 caption：{len(cases)} 条，覆盖 {len(imgids)} 张图")
    if cases:
        print("样例：", cases[0]["query"], "->", cases[0]["relevant_files"])

    # 语料图片列表：优先 ImageSets，否则用 caption 覆盖的图
    imgset = _find(args.split_dir, "ImageSets", "*.txt")
    if imgset:
        corpus_ids = [ln.strip() for ln in open(imgset, encoding="utf-8") if ln.strip()]
    else:
        corpus_ids = sorted(imgids)
    print(f"语料图片数：{len(corpus_ids)}")

    if args.images_src and args.out_images:
        src, out = Path(args.images_src), Path(args.out_images)
        out.mkdir(parents=True, exist_ok=True)
        copied_files: set[str] = set()
        missing = 0
        for iid in corpus_ids:
            p = src / f"{iid}.jpg"
            if p.exists():
                shutil.copy(p, out / f"{iid}.jpg")
                copied_files.add(f"{iid}.jpg")
            else:
                missing += 1
        print(f"复制图片 {len(copied_files)} 张 -> {out}（缺失 {missing}）")
        if missing:
            print("⚠️  有缺失图片：先补全原图（重跑下载可断点续传）再重跑本脚本更佳")
            # 评测集与语料保持一致：ground-truth 图不在语料里的用例必然无法命中，剔除
            before = len(cases)
            cases = [c for c in cases if all(f in copied_files for f in c["relevant_files"])]
            print(f"已剔除 ground-truth 缺失的用例 {before - len(cases)} 条，剩 {len(cases)} 条")

    out_eval = Path(args.out_eval)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测集 -> {out_eval}")


if __name__ == "__main__":
    main()
