#!/usr/bin/env python
"""全量索引脚本：等价于 `media-mind index --full`，方便 cron / systemd 直接调用。"""

from __future__ import annotations

import argparse

from media_mind.config import load_config
from media_mind.context import build_pipeline, build_search_context
from media_mind.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="全量重建索引")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    ctx = build_search_context(cfg)
    try:
        build_pipeline(ctx).run_full_index(cfg.data_sources)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
