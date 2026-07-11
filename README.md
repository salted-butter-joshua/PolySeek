<div align="center">

# 🔍 PolySeek

**自托管的多模态语义检索引擎**
一次索引，用自然语言检索你的图片、视频、音频与文档。

*One space for every modality — search your media by meaning, not filenames.*

[![CI](https://github.com/salted-butter-joshua/PolySeek/actions/workflows/ci.yml/badge.svg)](https://github.com/salted-butter-joshua/PolySeek/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

[快速开始](#-快速开始) · [功能](#-功能) · [使用](#-使用) · [配置](#-配置与技术栈切换) · [评测](#-合成数据与评测) · [架构](#-架构) · [路线图](#-路线图)

</div>

---

## 简介

PolySeek 把图片、视频关键帧、音频转写、文本文档统一编码进 **同一个 CLIP/SigLIP 语义向量空间**，
因此四种检索模式（文搜文 / 文搜图 / 图搜图 / 文搜视频 / 文搜音频）共享同一套索引与同一个查询入口。
适合部署在 NAS 或单卡服务器上，为个人/团队的媒体资产提供"用一句话找到那张图/那段视频"的能力。

> 为什么不用 VLM 生成 caption 再全文检索？向量检索延迟低 100x、无信息损失、资源省。
> 详见 [ADR-0001](docs/adr/0001-use-clip-not-vlm.md)。

## ✨ 功能

- 🔎 **五种检索模式**：文搜文、文搜图、图搜图、文搜视频（定位到帧时间戳）、文搜音频（定位到转写片段）
- 🔌 **可插拔 Embedding 后端**：Chinese-CLIP · SigLIP 2 · OpenAI CLIP，改配置即切换（[ADR-0004](docs/adr/0004-embedding-backend-abstraction.md)）
- 🗄️ **可插拔向量库**：Milvus Lite（嵌入式）· Qdrant（Docker），改配置即切换（[ADR-0005](docs/adr/0005-vector-store-milvus-vs-qdrant.md)）
- 📥 **多模态摄取**：图片直编码 · 视频 ffmpeg 抽帧 · 音频 Whisper 转写 · 文档切块
- ♻️ **增量索引**：xxhash 去重 + mtime 变更检测（SQLite 状态库），新增/修改/删除自动同步
- 🖥️ **三种接口**：CLI（Typer）· REST API（FastAPI）· Web UI（Gradio）
- ⏱️ **全链路计时**：查询编码 / 向量检索 / 离线分类型嵌入耗时，前端与 API 实时可见
- 🚀 **GPU + Docker**：RTX 20/30/40 系一键部署（[ADR-0008](docs/adr/0008-gpu-docker-deployment.md)）
- 🧪 **可复现评测**：内置合成数据生成器 + 100 条评测用例（Recall@K / MRR / 延迟）
- 🏗️ **工程化**：pydantic 强类型配置 · loguru 日志 · pytest（Fake 组件，秒级）· ruff · CI

## 🚀 快速开始

### Docker（推荐）

#### CPU（跨平台，含 Windows）

```bash
git clone https://github.com/salted-butter-joshua/PolySeek.git && cd PolySeek
# 把媒体放到 ./sample_data/{images,videos,audio,text}（或改 config.docker.yaml）
docker compose up -d qdrant api          # 起向量库 + API
docker compose run --rm indexer          # 跑一次索引
docker compose up -d webui               # Web UI: http://localhost:7860
# API 文档: http://localhost:8900/docs
```

#### GPU（RTX 2080 / CUDA）

```bash
# 前置：nvidia-container-toolkit，验证 GPU 可见
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

docker compose -f docker-compose.gpu.yaml up -d qdrant api
docker compose -f docker-compose.gpu.yaml run --rm indexer     # GPU 索引
docker compose -f docker-compose.gpu.yaml up -d webui
```

> 🇨🇳 国内加速：Dockerfile 已内置清华 PyPI / Debian 镜像与 `HF_ENDPOINT=hf-mirror.com`。
> Docker Hub 拉取慢请配 `registry-mirrors`（见 [FAQ](#-faq)）。

### pip 本地安装

```bash
pip install -e .                          # 核心依赖
pip install -e ".[siglip,audio,webui]"    # 可选：SigLIP / 音频转写 / Web UI
pip install cn-clip                        # Chinese-CLIP 后端
# 视频/音频需系统安装 ffmpeg

polyseek index --full
polyseek search "金色麦田 夕阳" --type image
```

## 📖 使用

### CLI

```bash
polyseek index --full                              # 全量索引
polyseek index                                     # 增量索引
polyseek search "红色的圆形" --type text            # 文搜文
polyseek search "海边日落" --type image             # 文搜图
polyseek search "猫在玩毛线球" --type video_frame    # 文搜视频
polyseek search "关于深度学习的讨论" --type audio_transcript  # 文搜音频
polyseek search "海边日落"                          # 混合（全类型）
polyseek similar /path/to/query.jpg                # 图搜图
polyseek stats                                     # 索引统计
polyseek serve                                     # 启动 REST API
```

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/api/health` | 健康检查 |
| `GET`  | `/api/stats` | 索引统计 |
| `GET`  | `/api/config` | 当前生效配置 |
| `GET`  | `/api/timing` | 离线嵌入耗时（分类型） |
| `GET`  | `/api/search/text?q=&top_k=&media_type=` | 文本检索（`media_type=text` 即文搜文） |
| `POST` | `/api/search/image` | 以图搜图（multipart: `file=@query.jpg`） |
| `GET`  | `/api/file?path=` | 读取结果文件 |

搜索响应内含 `timing`（`embed_ms` / `search_ms` / `total_ms`）。

### Web UI

`http://<host>:7860` —— 支持文本/图片检索、结果画廊、**前端热切换配置**（后端/模型/向量库/设备/batch），
并实时展示查询耗时与"📊 离线嵌入统计"（文本/图片/视频/音频分开）。

### 定时增量（crontab）

```cron
0 * * * * cd /app && python scripts/index_incremental.py >> /var/log/polyseek.log 2>&1
```

## ⚙️ 配置与技术栈切换

改 `config.yaml` 即可，无需改代码：

```yaml
embedding:
  backend: siglip                          # chinese_clip | siglip | openai_clip
  model_name: google/siglip2-base-patch16-224
  device: cuda                             # cpu | cuda | mps
vector_store:
  backend: qdrant                          # milvus_lite | qdrant
```

也支持环境变量覆盖（优先级：环境变量 > YAML > 默认值）：

```bash
export POLYSEEK__EMBEDDING__BACKEND=siglip
export POLYSEEK__EMBEDDING__DEVICE=cuda
```

> 换 Embedding 后端会改变向量维度，需重建索引（[ADR-0004](docs/adr/0004-embedding-backend-abstraction.md)）。

## 🧪 合成数据与评测

内置合成数据生成器（颜色×形状概念，文件名编码 ground truth）与 100 条评测用例：

```bash
# 1. 生成 ~2GB 数据（文本/图片/视频/音频）
python scripts/generate_data.py --out sample_data --target-gb 2.0

# 2. 建索引（记录分类型离线嵌入耗时到 data/index_stats.json）
polyseek index --full

# 3. 生成 100 条评测用例（文搜文/文搜图/图搜图各 ~1/3）
python scripts/generate_eval.py --manifest sample_data/manifest.json --out eval/dataset.json --n 100

# 4. 跑评测：各模式 Recall@1/5/10、MRR、查询编码/检索耗时
python scripts/run_eval.py --eval eval/dataset.json --top-k 10 --json eval/report.json
```

评测报告（示例格式）：

```
mode           n     R@1     R@5    R@10     MRR  embed_ms search_ms
--------------------------------------------------------------------
text2text     34   ...     ...     ...     ...      ...      ...
text2image    33   ...     ...     ...     ...      ...      ...
image2image   33   ...     ...     ...     ...      ...      ...
OVERALL      100   ...
```

命中判定按结果文件名解析的"概念"匹配，与索引绝对路径无关，跨环境稳定。

## 🏗️ 架构

```
              查询层         CLI (Typer) · REST API (FastAPI) · Web UI (Gradio)
                                          │  text / image query
              检索层         SearchEngine：编码 query → 过滤 → ANN 检索 → 去重排序
                                          │
              推理层         EmbeddingService（Chinese-CLIP / SigLIP2 / OpenAI CLIP）
                                          │  归一化向量 [d]
              存储层         VectorStore（Milvus Lite / Qdrant）+ metadata 过滤
                                          │
              摄取层         Scanner → {Image/Video/Audio/Text}Processor → Embedding → Store
                                          │  增量：xxhash + mtime（SQLite 状态库）
              数据层         NAS: images / videos / audio / text
```

四种模式共享同一嵌入空间与同一向量索引。设计取舍详见 [docs/adr/](docs/adr/)。

### 项目结构

```
polyseek/
├── polyseek/              # Python 包（导入名，发行名为 polyseek）
│   ├── config.py            # pydantic 强类型配置
│   ├── context.py           # 依赖装配（config→embedding→store→engine）
│   ├── embedding/           # base + chinese_clip + siglip + openai_clip + 工厂
│   ├── storage/             # base + milvus_store + qdrant_store + 工厂
│   ├── ingestion/           # scanner + image/video/audio/text processor + pipeline + stats
│   ├── search/engine.py     # 统一检索引擎（含带计时的 *_timed 方法）
│   ├── data_gen/            # 合成数据：concepts + shapes + text_gen
│   ├── api/ · cli/ · webui/ # 三种接口
├── scripts/                 # index_* / generate_data / generate_eval / run_eval / benchmark / export_onnx
├── tests/                   # pytest（Fake 组件，无需模型/DB）
├── docs/adr/                # 架构决策记录（ADR-0001..0008）
├── config.yaml · config.docker.yaml
├── Dockerfile · Dockerfile.gpu · docker-compose*.yaml
└── pyproject.toml · Makefile
```

## 📊 性能参考（CPU, ViT-B-16）

| 项目 | 估算 |
|------|------|
| 5 万图 + 500 视频(~15万帧) + 1000 音频 | ~25 万向量 |
| 向量 + HNSW 索引磁盘 | ~1.3 GB |
| 运行内存 | ~1.7 GB |
| 单次查询延迟 | ~55ms（编码 50ms + 检索 5ms） |
| ONNX 加速图像编码 | 2–3x |

索引选型：`<10万` 用 FLAT（recall 100%），`10~100万` 用 HNSW（recall ~98%，快 10x），`>100万` 用 IVF。

## 🛠️ 开发

```bash
pip install -e ".[dev]"
make lint     # ruff + mypy
make test     # pytest（快速，不依赖真实模型/向量库）
```

测试使用 `tests/conftest.py` 的 `FakeEmbedding` + `InMemoryStore`，秒级跑完，
覆盖配置校验、增量检测、检索去重、批量对齐、计时统计等核心逻辑。

常用 Make 目标：`make data`（生成数据）· `make eval` · `make run-eval` · `make serve` · `make ui` ·
`make docker-up` · `make gpu-up`。

**类型检查策略（渐进式）**：CI 中 `mypy` 为**非阻断**（`|| true`）——这是有意为之的
「渐进式类型化」决策，而非疏漏。核心模块已带类型标注，但尚未做到全量零告警；随着标注
补全，未来会去掉 `|| true` 转为强制。`ruff` 与 `pytest` 则是**强制门禁**，失败即挂 CI。

## 🗺️ 路线图

- [x] Phase 1：图片索引 + 文搜图 / 图搜图 + CLI
- [x] Phase 2：增量索引 + 视频抽帧 + 音频转写 + 文搜文 + REST API
- [x] Phase 3：SigLIP 后端 + ONNX 导出 + Gradio UI + 评测 + GPU/Docker
- [ ] 结果重排（rerank）与跨模态融合打分
- [ ] 场景切换检测抽帧、视频级 Embedding 可选升级
- [ ] 分布式 Milvus 迁移、鉴权与多租户

## ❓ FAQ

**Windows 本机能跑吗？** 能，但 Milvus Lite 仅支持 Linux/macOS，Windows 请把 `vector_store.backend`
设为 `qdrant`（Docker 起 Qdrant），或整套用 Docker。

**Docker Hub 拉取很慢？** 配置国内镜像加速器：

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{ "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1panel.live"] }
EOF
sudo systemctl restart docker
```

**合成音频能被文搜音频检索到吗？** 生成器产出的是正弦音（无语音内容），Whisper 不会产出有意义转写；
真实语音检索请用真实音频或 TTS 素材。

## 🤝 贡献

欢迎 Issue 与 PR。提交前请跑 `make lint && make test`。新增架构决策请在 `docs/adr/` 下按
[模板](docs/adr/0000-template.md) 记录。

## 📄 License

[MIT](LICENSE)

## 🙏 致谢

- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) · [SigLIP](https://github.com/google-research/big_vision) · [OpenAI CLIP](https://github.com/openai/CLIP)
- [Milvus](https://milvus.io/) · [Qdrant](https://qdrant.tech/) · [Whisper](https://github.com/openai/whisper)
- [FastAPI](https://fastapi.tiangolo.com/) · [Typer](https://typer.tiangolo.com/) · [Gradio](https://gradio.app/)
