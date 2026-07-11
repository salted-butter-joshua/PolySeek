# 架构决策记录（ADR）

本目录记录 media-mind 的关键架构与技术选型决策。每条 ADR 是一个不可变的历史快照：
决策一旦做出就记录下来，后续若被推翻，新增一条 ADR 并把旧的标记为 `Superseded`，而不是删改。

格式参考 [MADR](https://adr.github.io/madr/)。新建 ADR 请复制 [0000-template.md](0000-template.md)。

## 索引

| 编号 | 标题 | 状态 |
|------|------|------|
| [0001](0001-use-clip-not-vlm.md) | 用 CLIP 向量检索而非 VLM 生成 caption | Accepted |
| [0002](0002-frame-level-video-embedding.md) | 视频用帧级 CLIP 而非视频 Embedding 模型 | Accepted |
| [0003](0003-audio-whisper-not-clap.md) | 音频走 Whisper 转写而非 CLAP | Accepted |
| [0004](0004-embedding-backend-abstraction.md) | Embedding 后端抽象 + 可插拔（Chinese-CLIP/SigLIP2/OpenAI CLIP） | Accepted |
| [0005](0005-vector-store-milvus-vs-qdrant.md) | 向量库双后端：Milvus Lite + Qdrant | Accepted |
| [0006](0006-pydantic-strong-typed-config.md) | 强类型配置（pydantic）+ 环境变量覆盖 | Accepted |
| [0007](0007-text-to-text-chunking.md) | 文搜文：文档切块进同一 CLIP 空间 | Accepted |
| [0008](0008-gpu-docker-deployment.md) | GPU + Docker 部署（RTX 2080 / CUDA 12.1） | Accepted |

## 状态说明

- **Proposed**：提议中，尚未采纳
- **Accepted**：已采纳并实施
- **Superseded by ADR-XXXX**：已被新决策取代
- **Deprecated**：不再适用
