.PHONY: help install install-dev lint test index index-full search serve ui \
        data eval run-eval bench compare docker-up docker-index gpu-up gpu-index clean

CONFIG ?= config.yaml

help:
	@echo "make install       安装运行依赖（不含具体 CLIP 权重库，按需装 extras）"
	@echo "make install-dev   安装开发依赖"
	@echo "make lint          ruff + mypy 检查"
	@echo "make test          运行单元测试（无需模型/外部 DB）"
	@echo "make index         增量索引"
	@echo "make index-full    全量索引"
	@echo "make serve         启动 REST API"
	@echo "make ui            启动 Gradio Web UI"
	@echo "make docker-up     docker compose 起 qdrant + api"
	@echo "make docker-index  容器内跑一次索引"

install:
	pip install -e .

install-dev:
	pip install -e ".[siglip,audio,onnx,webui,dev]"

lint:
	ruff check polyseek tests
	mypy polyseek

test:
	pytest

index:
	python -m polyseek.cli.main index

index-full:
	python -m polyseek.cli.main index --full

serve:
	python -m polyseek.cli.main serve

ui:
	python -m polyseek.webui.app

data:
	python scripts/generate_data.py --out sample_data --target-gb 2.0

eval:
	python scripts/generate_eval.py --manifest sample_data/manifest.json --out eval/dataset.json --n 100

run-eval:
	python scripts/run_eval.py --eval eval/dataset.json --top-k 10 --json eval/report.json

# ③ 单后端 benchmark → 生成 docs/benchmark.md
bench:
	python scripts/run_eval.py --eval eval/dataset.json --top-k 10 -c $(CONFIG) --json eval/report.json
	python scripts/report_to_markdown.py --stats data/index_stats.json --report eval/report.json --out docs/benchmark.md

# ④ 多后端对比（Chinese-CLIP vs SigLIP）→ 生成 docs/benchmark.md
compare:
	bash scripts/compare_backends.sh $(CONFIG)

docker-up:
	docker compose up -d qdrant api

docker-index:
	docker compose run --rm indexer

gpu-up:
	docker compose -f docker-compose.gpu.yaml up -d qdrant api

gpu-index:
	docker compose -f docker-compose.gpu.yaml run --rm indexer

clean:
	rm -rf data/*.db data/*.db.* __pycache__ .pytest_cache .ruff_cache .mypy_cache
