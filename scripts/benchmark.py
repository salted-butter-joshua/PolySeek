#!/usr/bin/env python
"""检索质量评估：Recall@K 与 MRR。

评估集格式（JSON）：
[
  {"query": "一只橙色的猫", "relevant_paths": ["/nas/photos/cat1.jpg", "..."]},
  ...
]
relevant_paths 是该 query 的 ground-truth 命中文件（按文件路径匹配）。

用法：
    python scripts/benchmark.py --eval-set eval.json --top-k 10 --type image
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from polyseek.config import load_config
from polyseek.context import build_search_context
from polyseek.logging_setup import setup_logging


def evaluate(engine, eval_items: list[dict], top_k: int, media_type: str | None) -> dict:
    recall_at = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_total = 0.0
    latencies: list[float] = []

    for item in eval_items:
        query = item["query"]
        relevant = {str(Path(p)) for p in item["relevant_paths"]}

        t0 = time.perf_counter()
        results = engine.text_search(query, top_k=max(top_k, 10), media_type=media_type)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked = [str(Path(r.file_path)) for r in results]

        for k in recall_at:
            hit = any(p in relevant for p in ranked[:k])
            recall_at[k] += 1.0 if hit else 0.0

        rr = 0.0
        for rank, p in enumerate(ranked, 1):
            if p in relevant:
                rr = 1.0 / rank
                break
        mrr_total += rr

    n = max(1, len(eval_items))
    latencies.sort()
    return {
        "num_queries": len(eval_items),
        "recall@1": recall_at[1] / n,
        "recall@5": recall_at[5] / n,
        "recall@10": recall_at[10] / n,
        "mrr": mrr_total / n,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量基准测试")
    parser.add_argument("--eval-set", required=True, help="评估集 JSON 路径")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--type", dest="media_type", default="image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    with open(args.eval_set, encoding="utf-8") as f:
        eval_items = json.load(f)

    ctx = build_search_context(cfg, ensure_collection=False)
    try:
        metrics = evaluate(ctx.engine, eval_items, args.top_k, args.media_type)
    finally:
        ctx.close()

    print("\n===== Benchmark =====")
    print(f"backend={cfg.embedding.backend} model={cfg.embedding.model_name}")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
