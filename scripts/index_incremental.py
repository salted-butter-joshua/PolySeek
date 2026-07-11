#!/usr/bin/env python
"""增量索引脚本：等价于 `media-mind index`，适合 crontab 每小时跑一次。

    0 * * * * cd /app && python scripts/index_incremental.py >> /var/log/media-mind.log 2>&1
"""

from __future__ import annotations

import argparse

from media_mind.config import load_config
from media_mind.context import build_pipeline, build_search_context
from media_mind.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="增量索引")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg)
    try:
        build_pipeline(ctx).run_incremental_index(cfg.data_sources)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
