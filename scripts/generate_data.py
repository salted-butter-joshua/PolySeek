#!/usr/bin/env python
"""合成多模态数据生成器：文本 / 图片 / 视频 / 音频，可指定目标总大小（默认 2GB）。

设计：以"颜色×形状"为概念标签，文件名编码概念（``{颜色}_{形状}_{序号}.ext``），
使评测无需依赖绝对路径即可判定命中（按文件名解析概念）。图片作为体积主力（编码快，
适合度量吞吐），文本/视频/音频数量固定。

用法：
    python scripts/generate_data.py --out sample_data --target-gb 2.0
    python scripts/generate_data.py --out sample_data --target-gb 0.05 --no-video --no-audio
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 允许直接 `python scripts/generate_data.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyseek.data_gen.concepts import all_concepts  # noqa: E402
from polyseek.data_gen.shapes import draw_concept_image  # noqa: E402
from polyseek.data_gen.text_gen import generate_document  # noqa: E402


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fname(concept, idx: int, ext: str) -> str:
    return f"{concept.color_name}_{concept.shape_name}_{idx:05d}{ext}"


def gen_text(out: Path, docs: int, concepts, manifest: list) -> None:
    d = out / "text"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(docs):
        c = concepts[i % len(concepts)]
        path = d / _fname(c, i, ".md")
        path.write_text(generate_document(c, seed=i), encoding="utf-8")
        manifest.append(
            {"path": str(path), "modality": "text", "concept": c.key,
             "color": c.color_name, "shape": c.shape_name}
        )
    print(f"[text]  {docs} docs -> {d}")


def gen_images_until(out: Path, concepts, target_bytes: int, other_bytes: int,
                     img_size: int, manifest: list, num_images: int | None = None) -> None:
    d = out / "images"
    d.mkdir(parents=True, exist_ok=True)
    i = 0
    # 每 500 张才 stat 一次目录，避免大规模生成时 _dir_size 反复全量扫描拖慢
    size_check_every = 500
    running_bytes = other_bytes
    while True:
        if num_images is not None:
            if i >= num_images:
                break
        else:
            if i % size_check_every == 0:
                running_bytes = other_bytes + _dir_size(d)
            if running_bytes >= target_bytes:
                break
        c = concepts[i % len(concepts)]
        img = draw_concept_image(c.color_rgb, c.shape_key, size=img_size, seed=i)
        path = d / _fname(c, i, ".jpg")
        img.save(path, quality=88)
        manifest.append(
            {"path": str(path), "modality": "image", "concept": c.key,
             "color": c.color_name, "shape": c.shape_name}
        )
        i += 1
        if i % 500 == 0:
            tgt = f"/{num_images}" if num_images else f" ~{running_bytes/1e9:.2f} GB"
            print(f"[image] {i}{tgt} images ...")
    print(f"[image] {i} images -> {d}")


def gen_videos(out: Path, concepts, count: int, seconds: int, ffmpeg: str,
               img_size: int, manifest: list) -> None:
    d = out / "videos"
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(count):
            c = concepts[i % len(concepts)]
            frame = Path(tmp) / "frame.png"
            draw_concept_image(c.color_rgb, c.shape_key, size=img_size, seed=10_000 + i).save(frame)
            path = d / _fname(c, i, ".mp4")
            # 由概念静帧生成带轻微缩放的短视频；静态内容 h264 体积小，帧仍是概念画面
            cmd = [
                ffmpeg, "-y", "-v", "error", "-loop", "1", "-i", str(frame),
                "-t", str(seconds), "-r", "25",
                "-vf", f"scale={img_size}:{img_size},zoompan=z='min(zoom+0.001,1.2)':d=25*{seconds}:s={img_size}x{img_size}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ]
            subprocess.run(cmd, check=False)
            if path.exists():
                manifest.append(
                    {"path": str(path), "modality": "video", "concept": c.key,
                     "color": c.color_name, "shape": c.shape_name}
                )
    print(f"[video] {count} clips -> {d}")


def gen_audio(out: Path, concepts, count: int, seconds: int, ffmpeg: str,
              manifest: list) -> None:
    d = out / "audio"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        c = concepts[i % len(concepts)]
        freq = 220 + (i % 24) * 40  # 每概念确定性音高
        path = d / _fname(c, i, ".wav")
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
            "-ac", "1", "-ar", "16000", str(path),
        ]
        subprocess.run(cmd, check=False)
        if path.exists():
            manifest.append(
                {"path": str(path), "modality": "audio", "concept": c.key,
                 "color": c.color_name, "shape": c.shape_name}
            )
    print(f"[audio] {count} tones -> {d}  (注：正弦音无语音内容，Whisper 不会产出有意义转写)")


def main() -> None:
    p = argparse.ArgumentParser(description="生成合成多模态数据集")
    p.add_argument("--out", default="sample_data")
    p.add_argument("--target-gb", type=float, default=2.0)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--num-images", type=int, default=None,
                   help="精确生成的图片张数（设了则忽略 --target-gb 的图片填充，用于\"N万张图\"基准）")
    p.add_argument("--text-docs", type=int, default=240)
    p.add_argument("--videos", type=int, default=120)
    p.add_argument("--video-seconds", type=int, default=4)
    p.add_argument("--audios", type=int, default=80)
    p.add_argument("--audio-seconds", type=int, default=6)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--clean", action="store_true", help="先清空输出目录")
    args = p.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    concepts = all_concepts()
    manifest: list[dict] = []

    gen_text(out, args.text_docs, concepts, manifest)

    if not args.no_video and ffmpeg:
        gen_videos(out, concepts, args.videos, args.video_seconds, ffmpeg, args.img_size, manifest)
    elif not args.no_video:
        print("[video] ffmpeg 未找到，跳过视频生成")

    if not args.no_audio and ffmpeg:
        gen_audio(out, concepts, args.audios, args.audio_seconds, ffmpeg, manifest)
    elif not args.no_audio:
        print("[audio] ffmpeg 未找到，跳过音频生成")

    # 图片：按张数（--num-images）或按体积（--target-gb）填充
    target_bytes = int(args.target_gb * 1e9)
    other_bytes = _dir_size(out)  # 已生成的 text/video/audio
    if args.num_images is not None:
        gen_images_until(out, concepts, target_bytes, other_bytes, args.img_size,
                         manifest, num_images=args.num_images)
    elif other_bytes < target_bytes:
        gen_images_until(out, concepts, target_bytes, other_bytes, args.img_size, manifest)
    else:
        print("[image] 其它类型已达目标大小，跳过图片填充")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps({"concepts": [c.key for c in concepts], "files": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_gb = _dir_size(out) / 1e9
    by_mod: dict[str, int] = {}
    for m in manifest:
        by_mod[m["modality"]] = by_mod.get(m["modality"], 0) + 1
    print(f"\n完成：{total_gb:.2f} GB，共 {len(manifest)} 个文件 {by_mod}")
    print(f"清单：{manifest_path}")


if __name__ == "__main__":
    main()
