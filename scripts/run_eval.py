#!/usr/bin/env python
"""评测运行器：执行文搜文/文搜图/图搜图用例，输出 Recall@K、MRR 与耗时。

命中判定：结果文件名解析出的概念 == 用例目标概念（与绝对路径无关）。

用法：
    python scripts/run_eval.py --eval eval/dataset.json --top-k 10
    python scripts/run_eval.py --eval eval/dataset.json --json eval/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyseek.config import load_config  # noqa: E402
from polyseek.context import build_search_context  # noqa: E402
from polyseek.logging_setup import setup_logging  # noqa: E402


def concept_of(path: str) -> str:
    parts = Path(path).name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else ""


def _metrics(hits_at: dict, rr_sum: float, n: int) -> dict:
    n = max(1, n)
    return {
        "recall@1": hits_at[1] / n,
        "recall@5": hits_at[5] / n,
        "recall@10": hits_at[10] / n,
        "mrr": rr_sum / n,
    }


def run(engine, cases: list[dict], top_k: int) -> dict:
    per_mode: dict[str, dict] = defaultdict(
        lambda: {"hits": {1: 0, 5: 0, 10: 0}, "rr": 0.0, "n": 0,
                 "embed_ms": 0.0, "search_ms": 0.0}
    )
    overall = {"hits": {1: 0, 5: 0, 10: 0}, "rr": 0.0, "n": 0,
               "embed_ms": 0.0, "search_ms": 0.0}
    latencies: list[float] = []  # 每条 query 的总耗时（embed+search，ms）

    limit = max(top_k, 10)
    for case in cases:
        mode = case["mode"]
        target = case["target_concept"]
        mt = case["target_media_type"]

        if mode in ("text2text", "text2image"):
            ts = engine.text_search_timed(case["query"], top_k=limit, media_type=mt)
        elif mode == "image2image":
            ts = engine.image_search_timed(case["query_image"], top_k=limit, media_type=mt)
        else:
            continue

        ranked = [concept_of(r.file_path) for r in ts.results]
        latencies.append(ts.total_ms)

        for bucket in (per_mode[mode], overall):
            bucket["n"] += 1
            bucket["embed_ms"] += ts.embed_ms
            bucket["search_ms"] += ts.search_ms
            for k in (1, 5, 10):
                if target in ranked[:k]:
                    bucket["hits"][k] += 1
            for rank, c in enumerate(ranked, 1):
                if c == target:
                    bucket["rr"] += 1.0 / rank
                    break

    report = {"per_mode": {}, "overall": {}}
    for mode, b in per_mode.items():
        m = _metrics(b["hits"], b["rr"], b["n"])
        m.update({"n": b["n"], "avg_embed_ms": b["embed_ms"] / max(1, b["n"]),
                  "avg_search_ms": b["search_ms"] / max(1, b["n"])})
        report["per_mode"][mode] = m
    o = _metrics(overall["hits"], overall["rr"], overall["n"])
    lat = sorted(latencies)
    def _pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0.0
    o.update({"n": overall["n"], "avg_embed_ms": overall["embed_ms"] / max(1, overall["n"]),
              "avg_search_ms": overall["search_ms"] / max(1, overall["n"]),
              "latency_p50_ms": _pct(0.50), "latency_p95_ms": _pct(0.95),
              "latency_p99_ms": _pct(0.99)})
    report["overall"] = o
    return report


def _print(report: dict, cfg) -> None:
    print("\n===== Evaluation Report =====")
    print(f"backend={cfg.embedding.backend}  model={cfg.embedding.model_name}  "
          f"store={cfg.vector_store.backend}  device={cfg.embedding.device}")
    header = f"{'mode':<12}{'n':>4}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'embed_ms':>10}{'search_ms':>10}"
    print(header)
    print("-" * len(header))
    for mode, m in report["per_mode"].items():
        print(f"{mode:<12}{m['n']:>4}{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}"
              f"{m['recall@10']:>8.3f}{m['mrr']:>8.3f}{m['avg_embed_ms']:>10.2f}{m['avg_search_ms']:>10.2f}")
    o = report["overall"]
    print("-" * len(header))
    print(f"{'OVERALL':<12}{o['n']:>4}{o['recall@1']:>8.3f}{o['recall@5']:>8.3f}"
          f"{o['recall@10']:>8.3f}{o['mrr']:>8.3f}{o['avg_embed_ms']:>10.2f}{o['avg_search_ms']:>10.2f}")
    print(f"query 延迟 p50={o.get('latency_p50_ms', 0):.2f}  p95={o.get('latency_p95_ms', 0):.2f}  "
          f"p99={o.get('latency_p99_ms', 0):.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="运行评测用例")
    ap.add_argument("--eval", default="eval/dataset.json")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", default=None, help="把报告写入 JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    cases = json.loads(Path(args.eval).read_text(encoding="utf-8"))

    ctx = build_search_context(cfg, ensure_collection=False)
    try:
        report = run(ctx.engine, cases, args.top_k)
    finally:
        ctx.close()

    # 记录本次实验的配置，供报告/对比使用
    report["config"] = {
        "backend": cfg.embedding.backend,
        "model_name": cfg.embedding.model_name,
        "device": cfg.embedding.device,
        "vector_store": cfg.vector_store.backend,
        "dimension": ctx.embedding.dimension,
        "top_k": args.top_k,
    }
    _print(report, cfg)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.json}")


if __name__ == "__main__":
    main()
