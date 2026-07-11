# ADR-0008: GPU + Docker 部署（RTX 2080 / CUDA 12.1）

- **状态**：Accepted
- **日期**：2026-07-10

## 背景与问题

目标部署环境是一台 RTX 2080（Turing 架构）的 Ubuntu 服务器，需用 Docker 交付，并用 GPU
加速 Embedding 与 Whisper 转写。如何组织镜像与编排？

## 候选方案

- **方案 A**：`nvidia/cuda` 基础镜像 + 系统装 CUDA。
- **方案 B**：`python:3.11-slim` + pip 装 `torch` 的 cu121 wheels（自带 CUDA runtime）。

## 决策

采用方案 B：slim 镜像 + cu121 torch wheels，宿主机只需 NVIDIA 驱动 + nvidia-container-toolkit。
GPU 通过 compose 的 `deploy.resources.reservations.devices` 预留；`device=cuda` 由环境变量注入。

## 理由

1. cu121 wheels 自带 CUDA runtime，镜像更小、构建更简单，无需系统级 CUDA。
2. RTX 2080（Turing, SM 7.5）兼容 CUDA 12.1。
3. 向量库用 Qdrant（容器服务），规避 Milvus Lite 不支持容器/Windows 的问题（见 [0005](0005-vector-store-milvus-vs-qdrant.md)）。

## 后果

- **正面**：一条 `docker compose -f docker-compose.gpu.yaml up` 起全栈；CPU/GPU 仅镜像与
  `MEDIA_MIND__EMBEDDING__DEVICE` 之差。
- **负面**：镜像含完整 CUDA runtime（~3-4GB）；首次构建拉取较大。
- **前置条件**：`nvidia-container-toolkit` 必装，`docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi` 应可用。
- **显存参考**：ViT-B-16 + Whisper base 单卡 <4GB，RTX 2080（8GB）充裕，可增大 `batch_size`。
