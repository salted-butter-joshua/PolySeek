# ADR-0005: 向量库双后端 —— Milvus Lite + Qdrant

- **状态**：Accepted
- **日期**：2026-07-10

## 背景与问题

单机 NAS 场景要不要 Docker 起独立向量库？跨平台（含 Windows）怎么办？

## 候选方案

- **方案 A**：只用 FAISS（纯索引库）。
- **方案 B**：Milvus Lite（嵌入式单文件）。
- **方案 C**：Qdrant（Docker 服务）。
- **方案 D**：抽象 `VectorStore`，同时支持 Milvus Lite 与 Qdrant，config 切换。

## 决策

采用方案 D：抽象层 + 双后端。默认 Milvus Lite（本地零依赖），生产/跨平台用 Qdrant。

## 理由

1. FAISS 不管持久化、metadata 过滤、动态增删，需自己造轮子。
2. Milvus Lite 单 `.db` 文件、零运维，适合快速起步；但**仅支持 Linux/macOS**。
3. Qdrant 跨平台（含 Windows）、Docker 一键起、payload 过滤强，适合部署与横向扩展。
4. 抽象层让两者可切换，迁移无痛。

## 后果

- **正面**：本地开发用 Milvus Lite，Docker/GPU 部署用 Qdrant，一套代码两种形态。
- **负面**：要维护两套 store 实现和各自的过滤表达式翻译。
- **约束**：Windows 本机不能用 Milvus Lite，须切 Qdrant（见 [0008](0008-gpu-docker-deployment.md)）。

## 索引选型

- `< 10 万`：FLAT（recall 100%）
- `10~100 万`：HNSW（M=16, ef=200, recall ~98%，快 10x）
- `> 100 万`：IVF_FLAT / IVF_PQ（省内存）
