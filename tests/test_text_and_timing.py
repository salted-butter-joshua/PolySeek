"""文搜文（text 类型）摄取/检索 + 计时/统计的测试，全程用 Fake 组件。"""

from __future__ import annotations

from polyseek.config import AppConfig, DataSource, IndexingConfig
from polyseek.ingestion.pipeline import IngestionPipeline
from polyseek.ingestion.scanner import FileScanner
from polyseek.ingestion.text_processor import TextProcessor
from polyseek.search import SearchEngine


def test_text_processor_chunks(tmp_path):
    from polyseek.config import TextConfig

    doc = tmp_path / "a.txt"
    doc.write_text("这是一段足够长的中文文本，" * 30, encoding="utf-8")
    tp = TextProcessor(TextConfig(chunk_chars=50, chunk_overlap=10))
    chunks = tp.chunk_file(str(doc))
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 50 for c in chunks)
    assert all("start_char" in c and "end_char" in c for c in chunks)


def test_index_text_and_search(tmp_path, fake_embedding, memory_store):
    docs = tmp_path / "text"
    docs.mkdir()
    for i in range(3):
        (docs / f"doc_{i}.md").write_text(f"文档{i}的内容 " * 40, encoding="utf-8")

    cfg = AppConfig(
        indexing=IndexingConfig(
            state_db_path=str(tmp_path / "state.db"),
            stats_path=str(tmp_path / "stats.json"),
        ),
        data_sources=[DataSource(path=str(docs), media_types=["text"])],
    )
    scanner = FileScanner(cfg.indexing)
    pipeline = IngestionPipeline(fake_embedding, memory_store, scanner, cfg)
    pipeline.run_full_index(cfg.data_sources)

    assert memory_store.count() > 0  # 每篇文档切多块

    # 文搜文：media_type=text 只返回文本块
    engine = SearchEngine(fake_embedding, memory_store)
    results = engine.text_search("文档内容", top_k=5, media_type="text")
    assert all(r.media_type == "text" for r in results)

    # 统计落盘且记录了 text 类型
    from polyseek.ingestion.stats import IngestionStats

    stats = IngestionStats.load(cfg.indexing.stats_path)
    assert stats.per_type["text"].files == 3
    assert stats.per_type["text"].vectors > 0


def test_timed_search_reports_timing(fake_embedding, memory_store):
    import numpy as np

    vecs = np.stack([fake_embedding.encode_text(f"t{i}") for i in range(4)])
    memory_store.insert(
        vecs, [{"media_type": "text", "file_path": f"/d{i}.md", "mtime": 1} for i in range(4)]
    )
    engine = SearchEngine(fake_embedding, memory_store)
    ts = engine.text_search_timed("hello", top_k=3, media_type="text")
    assert ts.embed_ms >= 0 and ts.search_ms >= 0
    assert abs(ts.total_ms - (ts.embed_ms + ts.search_ms)) < 1e-6
    assert len(ts.results) == 3
