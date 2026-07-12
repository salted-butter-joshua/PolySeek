# 📊 基准测试

- **硬件**：RTX 2080 8GB · Ubuntu 22.04 · Docker（CUDA 12.x torch）
- **数据集**：Flickr30k-CN test（VisualSearch 人工翻译版）：**975 张真实图片语料 / 4875 条中文 query**
  （原始 test 集 1000 图，其中 25 张不在 HF 公开镜像 nlphuji/flickr30k 的 31014 张范围内，
  对应 125 条 query 已剔除，评测集与语料严格一致）
- **命中判定**：结果文件名与 ground-truth 图片精确匹配（caption ↔ image_id），跨环境可复现
- **向量库**：Qdrant v1.18（HNSW，M=16, ef_construct=200，COSINE）
- **公平性**：两后端共用同一评测集、同一批图、同一台机器，仅更换 Embedding；
  collection / 状态库 / 统计完全隔离
- **复现步骤**：[flickr30k-cn-experiment.md](flickr30k-cn-experiment.md)；
  原始报告：[reports/](reports/)

## A/B：Chinese-CLIP vs SigLIP 2（文搜图）

| 后端 | 模型 | 维度 | R@1 | R@5 | R@10 | MRR | 编码(ms) | p99(ms) |
|------|------|----:|----:|----:|-----:|----:|---------:|--------:|
| **chinese_clip** | ViT-B-16 | 512 | **0.625** | **0.867** | **0.926** | **0.729** | 6.5 | 9.1 |
| siglip2 | google/siglip2-base-patch16-224 | 768 | 0.493 | 0.761 | 0.839 | 0.607 | **4.1** | **7.0** |

**结论**（n=4875）：

1. **质量**：中文 query 上 Chinese-CLIP 全面领先——R@1 相对高 **26.8%**、R@5 +13.9%、
   R@10 +10.4%、MRR +20.1%。越靠前的指标差距越大，说明差距主要在精排能力。
   符合预期：SigLIP2 训练数据仅约 10% 为非英语，而 Chinese-CLIP 为中文原生训练。
2. **速度/成本**：SigLIP2 编码快 36%（4.1 vs 6.5 ms）、端到端 p99 低 24%（7.0 vs 9.1 ms），
   但 768d 向量需要 **1.5×** 的索引内存与磁盘。
3. **选型**：中文为主的检索场景默认 **Chinese-CLIP**（本项目默认后端，该决策由数据支撑）；
   多语言混合库或对编码吞吐敏感的场景可权衡 SigLIP2。

### Chinese-CLIP 基线明细

| 模式 | n | R@1 | R@5 | R@10 | MRR | 编码(ms) | 检索(ms) |
|------|--:|----:|----:|-----:|----:|---------:|---------:|
| text2image | 4875 | 0.625 | 0.867 | 0.926 | 0.729 | 6.48 | 2.00 |

端到端 query 延迟：p50 8.3 · p95 8.8 · **p99 9.1 ms**。
一句话：用一句中文描述检索 975 张真实图片，92.6% 的情况下正确图片进入 Top-10，P99 延迟 9.1ms。

### SigLIP2 明细（padding 修复后）

| 模式 | n | R@1 | R@5 | R@10 | MRR | 编码(ms) | 检索(ms) |
|------|--:|----:|----:|-----:|----:|---------:|---------:|
| text2image | 4875 | 0.493 | 0.761 | 0.839 | 0.607 | 4.12 | 2.12 |

端到端 query 延迟：p50 6.2 · p95 6.6 · **p99 7.0 ms**。

## 🐛 案例：A/B 评测抓出的 SigLIP padding bug（R@10 提升 23×）

首轮 A/B 中 SigLIP2 的 R@10 只有 **0.037**——975 张图随机猜 Top-10 命中率约 0.010，
即模型输出接近随机。排查发现：

- SigLIP 训练时文本**定长 padding 到 64 且不使用 attention mask**；
- 推理侧却用了动态 `padding=True`，文本嵌入偏离训练分布，图文空间错位；
- transformers 只给警告不报错，**量化指标是唯一的暴露途径**。

修复（`polyseek/embedding/siglip.py`）：统一 `padding="max_length"` 分词；同时移除
`last_hidden_state` 均值池化兜底（非对齐空间投影，静默使用会掩盖同类错位），改为显式报错。

**修复效果：R@10 0.037 → 0.839（23×）**。教训：跨模态检索的正确性无法靠"能跑通"验证，
必须有带 ground-truth 的量化评测兜底。

## 更新方式

重跑后用工具重新渲染：

```bash
python scripts/report_to_markdown.py \
  --report eval/report_cnclip.json --report eval/report_siglip.json \
  --hardware "RTX 2080 8GB, Ubuntu 22.04" --out docs/benchmark.md
```
