# syntax=docker/dockerfile:1

# ---- 基础镜像：含 ffmpeg（视频/音频处理需要） ----
# 国内加速：apt / pip / torch 走镜像源，可用 build-arg 覆盖为官方源。
FROM python:3.11-slim AS base

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    HF_HOME=/app/models \
    HF_ENDPOINT=https://hf-mirror.com \
    POLYSEEK_CONFIG=/app/config.docker.yaml

# apt 换国内镜像，再装 ffmpeg（视频抽帧 / whisper 音频解码）
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg patch build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 依赖层（先装依赖，利用缓存） ----
COPY pyproject.toml README.md ./
COPY polyseek ./polyseek

# CPU 版 torch 走交大镜像；其余依赖走清华 PyPI
RUN pip install --index-url ${TORCH_INDEX} torch torchvision \
    && pip install ".[siglip,audio,webui]"

# ---- 应用层 ----
COPY config.docker.yaml ./
COPY scripts ./scripts

# 数据/模型持久化目录（compose 里挂 volume）
RUN mkdir -p /app/data /app/models

EXPOSE 8900 7860

# 默认启动 REST API；索引/UI 通过 compose 的其他 service 或 docker exec 触发
CMD ["uvicorn", "polyseek.api.server:app", "--host", "0.0.0.0", "--port", "8900"]
