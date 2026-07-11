#!/usr/bin/env bash
# ④ 多后端对比实验：对每个 Embedding 后端各建一份索引并评测，最后汇总成 Markdown。
#
# 关键隔离（否则实验互相污染）：
#   - collection_name  按后端区分（不同后端向量维度不同，必须分表）
#   - state_db_path    按后端区分（增量索引按路径去重，不分开第二个后端会“无新文件”跳过）
#   - stats_path       按后端区分（离线嵌入耗时分开统计）
#
# 前置：数据已生成（sample_data/）、评测集已生成（eval/dataset.json）、qdrant 已启动。
#
# 用法（在装了 polyseek 的环境里，或容器内运行）：
#   bash scripts/compare_backends.sh config.docker.yaml
# 容器内：
#   docker compose -f docker-compose.gpu.yaml run --rm indexer bash scripts/compare_backends.sh /app/config.docker.yaml
set -euo pipefail

CONFIG="${1:-config.yaml}"
EVAL="${EVAL:-eval/dataset.json}"
TOPK="${TOPK:-10}"

# 要对比的后端： "标签|backend|model_name"
BACKENDS=(
  "cnclip|chinese_clip|ViT-B-16"
  "siglip|siglip|google/siglip2-base-patch16-224"
  # "openai|openai_clip|openai/clip-vit-base-patch16"   # 需要英文 query 才公平，默认注释
)

mkdir -p eval data
REPORTS=()

for spec in "${BACKENDS[@]}"; do
  IFS='|' read -r tag backend model <<< "$spec"
  echo "==================== [$tag] backend=$backend model=$model ===================="

  export POLYSEEK__EMBEDDING__BACKEND="$backend"
  export POLYSEEK__EMBEDDING__MODEL_NAME="$model"
  export POLYSEEK__VECTOR_STORE__QDRANT__COLLECTION_NAME="polyseek_${tag}"
  export POLYSEEK__VECTOR_STORE__MILVUS__COLLECTION_NAME="polyseek_${tag}"
  export POLYSEEK__INDEXING__STATE_DB_PATH="data/state_${tag}.db"
  export POLYSEEK__INDEXING__STATS_PATH="data/stats_${tag}.json"

  echo "[$tag] 建索引 ..."
  polyseek index --full -c "$CONFIG"

  echo "[$tag] 评测 ..."
  python scripts/run_eval.py --eval "$EVAL" --top-k "$TOPK" -c "$CONFIG" \
      --json "eval/report_${tag}.json"

  REPORTS+=("--report" "eval/report_${tag}.json")
done

echo "==================== 汇总对比 ===================="
python scripts/report_to_markdown.py "${REPORTS[@]}" \
  --hardware "${HARDWARE:-RTX 2080 8GB, Ubuntu 22.04}" \
  --out docs/benchmark.md

echo "对比报告已写入 docs/benchmark.md"
