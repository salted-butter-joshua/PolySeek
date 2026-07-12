# 📊 基准测试

- **硬件**：RTX 2080 8GB · Ubuntu 22.04 · Docker（CUDA 12.x torch）
- **数据集**：Flickr30k-CN test（VisualSearch 人工翻译版）：**975 张真实图片语料 / 4875 条中文 query**
  （原始 test 集 1000 图，其中 25 张不在 HF 公开镜像 nlphuji/flickr30k 的 31014 张范围内，
  对应 125 条 query 已剔除，评测集与语料严格一致）
- **命中判定**：结果文件名与 ground-truth 图片精确匹配（caption ↔ image_id），跨环境可复现
- **向量库**：Qdrant v1.18（HNSW，M=16, ef_construct=200，COSINE）
- **复现步骤**：[flickr30k-cn-experiment.md](flickr30k-cn-experiment.md)

## Chinese-CLIP 基线（文搜图）

Chinese-CLIP ViT-B-16（512d，GPU 编码）：

| 模式 | n | R@1 | R@5 | R@10 | MRR | 编码(ms) | 检索(ms) |
|------|--:|----:|----:|-----:|----:|---------:|---------:|
| text2image | 4875 | 0.625 | 0.867 | **0.926** | 0.729 | 6.28 | 1.99 |

端到端 query 延迟：**p50 8.2 ms · p95 8.6 ms · p99 9.1 ms**

一句话：用一句中文描述检索 975 张真实图片，92.6% 的情况下正确图片进入 Top-10，P99 延迟 9.1ms。

## Chinese-CLIP vs SigLIP2 A/B（进行中）

| 后端 | 模型 | 维度 | R@1 | R@5 | R@10 | MRR | 编码(ms) | p99(ms) |
|------|------|----:|----:|----:|-----:|----:|---------:|--------:|
| chinese_clip | ViT-B-16 | 512 | 0.625 | 0.867 | 0.926 | 0.729 | 6.28 | 9.1 |
| siglip2 | google/siglip2-base-patch16-224 | 768 | — | — | — | — | — | 待重测 |

> SigLIP2 首轮数据因下述 bug 无效（R@10=0.037≈随机），修复后待重跑
> （`bash scripts/compare_backends.sh`，两后端共用同一评测集与语料）。

## 🐛 案例：A/B 评测抓出的 SigLIP padding bug

首轮 A/B 中 SigLIP2 的 R@10 只有 **0.037**——975 张图随机猜 Top-10 命中率约 0.010，
即模型输出接近随机。排查发现：

- SigLIP 训练时文本**定长 padding 到 64 且不使用 attention mask**；
- 推理侧却用了动态 `padding=True`，文本嵌入偏离训练分布，图文空间错位；
- transformers 其实给了警告（`use padding='max_length'`），但不会报错，**指标是唯一的暴露途径**。

修复（`polyseek/embedding/siglip.py`）：统一 `padding="max_length"` 分词；同时移除了
`last_hidden_state` 均值池化兜底——那不是对齐空间的投影输出，静默使用会掩盖同类错位，
现在直接报错。

**教训**：跨模态检索的正确性无法靠"能跑通"验证，必须有带 ground-truth 的量化评测兜底。

## 更新方式

重跑后用工具重新渲染并替换本文件相应表格：

```bash
python scripts/report_to_markdown.py \
  --report eval/report_cnclip.json --report eval/report_siglip.json \
  --hardware "RTX 2080 8GB, Ubuntu 22.04" --out docs/benchmark.md
```
