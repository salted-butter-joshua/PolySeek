"""摄取管道与扫描器测试（图片路径 + 增量检测），全程用 Fake 组件。"""

from __future__ import annotations

from PIL import Image

from polyseek.config import AppConfig, DataSource, IndexingConfig
from polyseek.ingestion.pipeline import IngestionPipeline
from polyseek.ingestion.scanner import FileScanner


def _make_images(dirpath, n=3):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (16, 16), color=(i * 20, 0, 0))
        img.save(dirpath / f"img_{i}.jpg")


def test_scanner_detects_new_and_deleted(tmp_path):
    photos = tmp_path / "photos"
    _make_images(photos, 3)
    scanner = FileScanner(IndexingConfig(state_db_path=str(tmp_path / "state.db")))

    files = scanner.scan_directory(str(photos))
    assert len(files) == 3

    new, mod, deleted = scanner.detect_changes(files)
    assert len(new) == 3 and not mod and not deleted

    # 标记已索引后，再次检测应为空
    for f in new:
        scanner.mark_indexed(f, ["vid-" + f.file_hash])
    new2, mod2, del2 = scanner.detect_changes(scanner.scan_directory(str(photos)))
    assert not new2 and not mod2 and not del2

    # 删除一个文件后应检测到 deleted
    (photos / "img_0.jpg").unlink()
    _, _, del3 = scanner.detect_changes(scanner.scan_directory(str(photos)))
    assert len(del3) == 1


def test_pipeline_indexes_images(tmp_path, fake_embedding, memory_store):
    photos = tmp_path / "photos"
    _make_images(photos, 4)

    cfg = AppConfig(
        indexing=IndexingConfig(state_db_path=str(tmp_path / "state.db")),
        data_sources=[DataSource(path=str(photos), media_types=["image"])],
    )
    scanner = FileScanner(cfg.indexing)
    pipeline = IngestionPipeline(fake_embedding, memory_store, scanner, cfg)

    pipeline.run_full_index(cfg.data_sources)
    assert memory_store.count() == 4

    # 增量：无变化应不新增
    pipeline.run_incremental_index(cfg.data_sources)
    assert memory_store.count() == 4

    # 删除一个文件后增量应减少
    (photos / "img_0.jpg").unlink()
    pipeline.run_incremental_index(cfg.data_sources)
    assert memory_store.count() == 3
