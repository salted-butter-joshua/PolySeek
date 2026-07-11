"""media-mind：NAS 多模态检索引擎。

四种检索模式（文搜图 / 图搜图 / 文搜视频 / 文搜音频）共享同一个 CLIP/SigLIP
嵌入空间与同一套向量索引。
"""

__version__ = "0.1.0"

from .config import AppConfig, load_config

__all__ = ["AppConfig", "load_config", "__version__"]
