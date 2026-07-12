#!/usr/bin/env bash
# ④ 多后端对比实验：对每个 Embedding 后端各建一份索引并评测，最后汇总成 Markdown。
#
# 关键隔离（否则实验互相污染）：
#   - collection_name  按后端区分（不同后端向量维度不同，必须分表）
#   - state_db_path    按后端区分（增量索引按路径去重，不分开第二个后端会“无新文件”跳过）
#   - stats_path       按后端区分（离线嵌入耗时分开统计）
#
# 可配置环境变量：
#   EVAL      评测集 JSON（默认 eval/dataset.json）
#   TOPK      Top-K（默认 10）
#   OUT_DIR   结果输出根目录（默认仓库根）。容器里跑请指到挂载目录，
#             否则 --rm 后 report/benchmark 全部丢失！例如 OUT_DIR=/fk/results
#   HARDWARE  硬件描述（写进报告）
#
# 用法（本机）：
#   bash scripts/compare_backends.sh config.yaml
# 容器（推荐，结果落到宿主机挂载）：
#   docker compose -f docker-compose.gpu.yaml run --rm \
#     -v /data/flickr30k-cn:/fk \
#     -e EVAL=/fk/flickr30k_cn.json -e OUT_DIR=/fk/results \
#     indexer bash scripts/compare_backends.sh /app/config.docker.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-$ROOT_DIR/config.yaml}"
EVAL="${EVAL:-$ROOT_DIR/eval/dataset.json}"
TOPK="${TOPK:-10}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR}"

# 要对比的后端： "标签|backend|model_name"
BACKENDS=(
  "cnclip|chinese_clip|ViT-B-16"
  "siglip|siglip|google/siglip2-base-patch16-224"
  # "openai|openai_clip|openai/clip-vit-base-patch16"   # 英文基线，中文 query 上不公平，默认注释
)

mkdir -p "$OUT_DIR/eval" "$OUT_DIR/data" "$OUT_DIR/docs"
REPORTS=()

for spec in "${BACKENDS[@]}"; do
  IFS='|' read -r tag backend model <<< "$spec"
  echo "==================== [$tag] backend=$backend model=$model ===================="

  export POLYSEEK__EMBEDDING__BACKEND="$backend"
  export POLYSEEK__EMBEDDING__MODEL_NAME="$model"
  export POLYSEEK__VECTOR_STORE__QDRANT__COLLECTION_NAME="polyseek_${tag}"
  export POLYSEEK__VECTOR_STORE__MILVUS__COLLECTION_NAME="polyseek_${tag}"
  export POLYSEEK__INDEXING__STATE_DB_PATH="$OUT_DIR/data/state_${tag}.db"
  export POLYSEEK__INDEXING__STATS_PATH="$OUT_DIR/data/stats_${tag}.json"

  echo "[$tag] 建索引 ..."
  polyseek index --full -c "$CONFIG"

  echo "[$tag] 评测 ..."
  python "$SCRIPT_DIR/run_eval.py" --eval "$EVAL" --top-k "$TOPK" -c "$CONFIG" \
      --json "$OUT_DIR/eval/report_${tag}.json"

  REPORTS+=("--report" "$OUT_DIR/eval/report_${tag}.json")
done

echo "==================== 汇总对比 ===================="
python "$SCRIPT_DIR/report_to_markdown.py" "${REPORTS[@]}" \
  --hardware "${HARDWARE:-RTX 2080 8GB, Ubuntu 22.04}" \
  --out "$OUT_DIR/docs/benchmark.md"

echo "对比报告已写入 $OUT_DIR/docs/benchmark.md"
