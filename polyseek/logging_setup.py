"""统一日志配置（基于 loguru）。

在任意入口（CLI / API / 脚本）启动时调用一次 :func:`setup_logging`，
使全项目日志格式、级别、落盘策略一致。
"""

from __future__ import annotations

import sys

from loguru import logger

from .config import LoggingConfig

_CONFIGURED = False


def setup_logging(config: LoggingConfig) -> None:
    """根据配置初始化 loguru（幂等，多次调用只生效第一次的 sink 重置）。"""
    global _CONFIGURED
    logger.remove()  # 清掉 loguru 默认 handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=config.level, format=fmt, enqueue=True)

    if config.file:
        logger.add(
            config.file,
            level=config.level,
            format=fmt,
            rotation="50 MB",
            retention="14 days",
            compression="zip",
            enqueue=True,
        )

    _CONFIGURED = True
    logger.debug("Logging initialized at level {}", config.level)
