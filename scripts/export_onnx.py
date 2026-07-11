#!/usr/bin/env python
"""导出 Chinese-CLIP 视觉编码器到 ONNX（CPU 场景推理提速 2-3x）。

用法：
    python scripts/export_onnx.py --model ViT-B-16 --out models/clip_visual.onnx

导出后可用 onnxruntime 加载 clip_visual.onnx 做图像编码；文本编码建议仍走 PyTorch
（tokenizer 依赖较重，且文本编码通常不是瓶颈）。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CLIP 视觉编码器到 ONNX")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--out", default="models/clip_visual.onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default="./models")
    args = parser.parse_args()

    import torch
    from cn_clip.clip import load_from_name

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, _ = load_from_name(args.model, device=args.device, download_root=args.cache_dir)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=args.device)
    torch.onnx.export(
        model.visual,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported visual encoder to {args.out}")

    # 简单校验
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy.cpu().numpy()})
        print(f"ONNX runtime OK, output shape: {out[0].shape}")
    except ImportError:
        print("onnxruntime 未安装，跳过校验（pip install 'polyseek[onnx]'）")


if __name__ == "__main__":
    main()
