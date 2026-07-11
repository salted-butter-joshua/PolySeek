# 📊 基准测试

> 本文件由工具自动生成，尚未填充真实数据。

在部署环境（如 RTX 2080）上运行以下命令即可生成/覆盖本文件：

```bash
make data && make index-full && make eval

# ③ 单后端 benchmark（Recall@K / MRR / 延迟 p50·p95 + 离线分类型嵌入耗时）
make bench

# ④ 多后端对比（Chinese-CLIP vs SigLIP）
make compare
```

生成内容包括：

- **离线嵌入耗时**（文本 / 图片 / 视频 / 音频分开，含吞吐 向量/秒）
- **检索质量**（文搜文 / 文搜图 / 图搜图 的 Recall@1/5/10、MRR）
- **查询延迟**（编码 ms / 检索 ms / p50 / p95）
- 多后端下的横向对比表
