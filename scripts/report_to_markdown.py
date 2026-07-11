#!/usr/bin/env python
"""把 run_eval 的报告 + 离线嵌入统计渲染成 README 可直接粘贴的 Markdown。

单后端（③ benchmark）：
    python scripts/report_to_markdown.py --stats data/index_stats.json \\
        --report eval/report.json --hardware "RTX 2080 8GB" --out docs/benchmark.md

多后端对比（④ SigLIP 对比）：
    python scripts/report_to_markdown.py \\
        --report eval/report_cnclip.json --report eval/report_siglip.json \\
        --hardware "RTX 2080 8GB" --out docs/benchmark.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，打印含 emoji 的 Markdown 会报错；统一按 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _offline_table(stats: dict) -> str:
    lines = [
        "| 类型 | 文件数 | 向量数 | 嵌入耗时(s) | 预处理耗时(s) | 向量/秒 |",
        "|------|-------:|-------:|-----------:|-------------:|--------:|",
    ]
    total_embed = 0.0
    for t in ("text", "image", "video", "audio"):
        st = stats.get("per_type", {}).get(t)
        if not st:
            continue
        vps = st["vectors"] / st["embed_seconds"] if st["embed_seconds"] > 0 else 0.0
        total_embed += st["embed_seconds"]
        lines.append(
            f"| {t} | {st['files']} | {st['vectors']} | {st['embed_seconds']:.1f} | "
            f"{st['process_seconds']:.1f} | {vps:.1f} |"
        )
    lines.append(
        f"| **合计** |  |  | **{total_embed:.1f}** | | 墙钟 "
        f"{stats.get('total_wall_seconds', 0):.1f}s |"
    )
    return "\n".join(lines)


def _quality_table(report: dict) -> str:
    lines = [
        "| 模式 | n | R@1 | R@5 | R@10 | MRR | 编码(ms) | 检索(ms) |",
        "|------|--:|----:|----:|-----:|----:|---------:|---------:|",
    ]
    for mode, m in report.get("per_mode", {}).items():
        lines.append(
            f"| {mode} | {m['n']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['mrr']:.3f} | {m['avg_embed_ms']:.2f} | "
            f"{m['avg_search_ms']:.2f} |"
        )
    o = report.get("overall", {})
    lines.append(
        f"| **总计** | {o.get('n', 0)} | {o.get('recall@1', 0):.3f} | "
        f"{o.get('recall@5', 0):.3f} | {o.get('recall@10', 0):.3f} | {o.get('mrr', 0):.3f} | "
        f"{o.get('avg_embed_ms', 0):.2f} | {o.get('avg_search_ms', 0):.2f} |"
    )
    return "\n".join(lines)


def _compare_table(reports: list[dict]) -> str:
    """多后端 OVERALL 对比。"""
    lines = [
        "| 后端 | 模型 | 维度 | R@1 | R@5 | R@10 | MRR | 编码(ms) | p95(ms) |",
        "|------|------|----:|----:|----:|-----:|----:|---------:|--------:|",
    ]
    for r in reports:
        c = r.get("config", {})
        o = r.get("overall", {})
        lines.append(
            f"| {c.get('backend', '?')} | {c.get('model_name', '?')} | "
            f"{c.get('dimension', '?')} | {o.get('recall@1', 0):.3f} | {o.get('recall@5', 0):.3f} | "
            f"{o.get('recall@10', 0):.3f} | {o.get('mrr', 0):.3f} | {o.get('avg_embed_ms', 0):.2f} | "
            f"{o.get('latency_p95_ms', 0):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染 benchmark Markdown")
    ap.add_argument("--report", action="append", required=True, help="run_eval 的 report.json，可多次")
    ap.add_argument("--stats", default=None, help="index_stats.json（离线嵌入耗时）")
    ap.add_argument("--hardware", default="（填写硬件，如 RTX 2080 8GB, Ubuntu 22.04）")
    ap.add_argument("--dataset", default="合成数据集（颜色×形状概念）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reports = [_load(p) for p in args.report]
    md: list[str] = ["## 📊 基准测试", "", f"- **硬件**：{args.hardware}",
                     f"- **数据集**：{args.dataset}", ""]

    if args.stats:
        md += ["### 离线嵌入耗时（分类型）", "", _offline_table(_load(args.stats)), ""]

    if len(reports) == 1:
        md += ["### 检索质量与延迟", "", _quality_table(reports[0]), ""]
        o = reports[0].get("overall", {})
        md.append(
            f"> query 延迟 p50 {o.get('latency_p50_ms', 0):.2f} ms · "
            f"p95 {o.get('latency_p95_ms', 0):.2f} ms"
        )
    else:
        md += ["### 后端对比（OVERALL）", "", _compare_table(reports), "",
               "各后端明细："]
        for r in reports:
            b = r.get("config", {}).get("backend", "?")
            md += [f"\n**{b}**", "", _quality_table(r)]

    text = "\n".join(md) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
