# Flickr30k-CN 真实数据实验（基准 + Chinese-CLIP vs SigLIP2 A/B）

在 RTX 2080 Ubuntu 服务器上，用真实中文图文数据集跑出两组数字：

- **实验一（质量基线 + 延迟）**：Chinese-CLIP 在 Flickr30k-CN test 上的 Recall@1/5/10、MRR、P50/P95/P99
- **实验二（A/B）**：同一评测集下 Chinese-CLIP(512d) vs SigLIP2(768d) 的对比表

> 方法论：合成数据（`generate_data.py`）测吞吐/延迟规模；**公开标注数据测检索质量**。
> 命中判定按 ground-truth 文件精确匹配（caption ↔ 图片 id），可复现。

## 数据组成（两部分，缺一不可）

| 部分 | 内容 | 来源 |
|------|------|------|
| 中文标注包 | `flickr30k-cn.tar.gz`（VisualSearch 格式：ImageSets + 分词 caption；**不含图片**） | lixirong/VisualSearch |
| 原始图片 | Flickr30k 全量 31,783 张 `<imgid>.jpg` | Kaggle / HuggingFace（见下） |

标注包解压后用 `flickr30kzhmbosontest`（人工翻译 test 集，约 1000 图 × 5 caption）。

## 第 0 步：权限 + 代码

```bash
sudo chown -R $USER: /data/flickr30k-cn
cd /data/multi-model-retrieval/media-mind && git pull
```

## 第 1 步：下载原始图片（选一种）

**A. Kaggle（能直连时最简单）**

```bash
pip install kaggle   # 需 ~/.kaggle/kaggle.json API token
kaggle datasets download -d hsankesara/flickr-image-dataset -p /data
unzip -q /data/flickr-image-dataset.zip -d /data
mv /data/flickr30k_images/flickr30k_images /data/flickr30k-images
```

**B. HuggingFace 镜像（国内推荐，走 hf-mirror，支持断点续传）**

```bash
mkdir -p /data/flickr30k-images
docker compose -f docker-compose.gpu.yaml run --rm \
  -v /data/flickr30k-images:/raw indexer sh -c \
  'pip install -q datasets && python scripts/download_flickr30k_images.py --out /raw'
```

完成后 `ls /data/flickr30k-images | wc -l` 应为 31783。

## 第 2 步：解析标注 + 筛出 test 语料图 + 生成评测集

```bash
cd /data/multi-model-retrieval/media-mind
docker compose -f docker-compose.gpu.yaml run --rm \
  -v /data/flickr30k-cn:/fk -v /data/flickr30k-images:/raw \
  indexer python scripts/prepare_flickr30k_cn.py \
    --split-dir /fk/flickr30kzhmbosontest \
    --images-src /raw \
    --out-images /fk/test-images/images \
    --out-eval /fk/flickr30k_cn.json
```

- `--out-images` 落在 `test-images/images/`（子目录 `images` 对应 config 数据源 `/data/media/images`）
- 评测集写到宿主机 `/data/flickr30k-cn/flickr30k_cn.json`（不放容器卷，防止清卷时丢失）
- 检查输出：`复制图片 ~1000 张（缺失 0）`、`解析 caption ~5000 条`，样例中文通顺

## 第 3 步：清库 + 索引 test 语料

```bash
export POLYSEEK_MEDIA_DIR=/data/flickr30k-cn/test-images   # 每个新终端都要 export

docker compose -f docker-compose.gpu.yaml down
docker volume rm media-mind_qdrant_storage media-mind_index_state
docker compose -f docker-compose.gpu.yaml up -d --force-recreate qdrant
docker compose -f docker-compose.gpu.yaml run --rm indexer
```

结束时记录日志里的 `Total vectors`（应 ≈ 图片数）。离线嵌入耗时在卷内 `/app/data/index_stats.json`。

## 第 4 步：实验一 —— Chinese-CLIP 基线

```bash
docker compose -f docker-compose.gpu.yaml run --rm \
  -v /data/flickr30k-cn:/fk indexer \
  python scripts/run_eval.py --eval /fk/flickr30k_cn.json \
    -c /app/config.docker.yaml --json /fk/report_cnclip.json
```

终端打印 text2image 的 R@1/5/10、MRR、以及 p50/p95/**p99**。可再渲染单后端报告：

```bash
docker compose -f docker-compose.gpu.yaml run --rm \
  -v /data/flickr30k-cn:/fk indexer \
  python scripts/report_to_markdown.py \
    --stats /app/data/index_stats.json --report /fk/report_cnclip.json \
    --hardware "RTX 2080 8GB, Ubuntu 22.04" \
    --dataset "Flickr30k-CN test (1000 图 / ~5000 中文 query)" \
    --out /fk/results/benchmark_cnclip.md
```

## 第 5 步：实验二 —— Chinese-CLIP vs SigLIP2 A/B

```bash
export POLYSEEK_MEDIA_DIR=/data/flickr30k-cn/test-images
docker compose -f docker-compose.gpu.yaml run --rm \
  -v /data/flickr30k-cn:/fk \
  -e EVAL=/fk/flickr30k_cn.json \
  -e OUT_DIR=/fk/results \
  -e HARDWARE="RTX 2080 8GB, Ubuntu 22.04" \
  indexer bash scripts/compare_backends.sh /app/config.docker.yaml
```

脚本自动：cnclip 建索引(`polyseek_cnclip`)→评测 → siglip 建索引(`polyseek_siglip`)→评测 → 汇总。
两个后端**共用同一份评测集与同一批图**，只换 embedding；collection / state DB / stats 全部隔离。

> SigLIP2 权重从 HF 镜像（hf-mirror）下载，首次约 1-2GB。若 `google/siglip2-base-patch16-224`
> 拉不下来，可退回多语言 v1：`google/siglip-base-patch16-256-multilingual`（改脚本 BACKENDS 行）。

## 第 6 步：读结果

产物全在宿主机 `/data/flickr30k-cn/results/`：

```
results/eval/report_cnclip.json    # 原始指标（含 config、p99）
results/eval/report_siglip.json
results/docs/benchmark.md          # 对比表（README 可直接粘）
```

对比表形如：

```
| 后端 | 模型 | 维度 | R@1 | R@5 | R@10 | MRR | 编码(ms) | p95(ms) |
| chinese_clip | ViT-B-16     | 512 | ... | ... | ... | ... | ... | ... |
| siglip       | siglip2-base | 768 | ... | ... | ... | ... | ... | ... |
```

提升幅度 = `(siglip_R@10 − cnclip_R@10) / cnclip_R@10`。把表贴进仓库 `docs/benchmark.md`
和 README 性能小节即可。

## 常见坑

- **忘了 `export POLYSEEK_MEDIA_DIR`** → 挂到空的 `./sample_data`，护栏会中止但索引不了
- **图片没放 `images/` 子目录** → 数据源是 `/data/media/images`，扫不到
- **OUT_DIR 没指到挂载目录** → 容器 `--rm` 后 report/benchmark 丢失
- **两后端复用同一 collection** → 维度 512 vs 768 冲突（脚本已隔离，别手动改 collection）
