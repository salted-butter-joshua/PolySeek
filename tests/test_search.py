"""检索引擎测试：命中、过滤、视频去重、混合搜索。"""

from __future__ import annotations

import numpy as np

from polyseek.search import SearchEngine


def _seed_store(store, embedding):
    """插入若干图片 + 同一视频的多帧。"""
    # 图片
    img_vecs = np.stack([embedding.encode_image(f"/photos/p{i}.jpg") for i in range(5)])
    img_meta = [
        {"media_type": "image", "file_path": f"/photos/p{i}.jpg", "mtime": 1000 + i}
        for i in range(5)
    ]
    store.insert(img_vecs, img_meta)

    # 同一视频的 3 帧，向量各不同
    vid_vecs = np.stack([embedding.encode_image(f"/videos/v.mp4::{ts}") for ts in (0, 2, 4)])
    vid_meta = [
        {"media_type": "video_frame", "file_path": "/videos/v.mp4", "mtime": 2000, "frame_ts": ts}
        for ts in (0, 2, 4)
    ]
    store.insert(vid_vecs, vid_meta)


def test_text_search_returns_results(memory_store, fake_embedding):
    _seed_store(memory_store, fake_embedding)
    engine = SearchEngine(fake_embedding, memory_store)
    results = engine.text_search("hello", top_k=3)
    assert len(results) == 3
    # 结果按 score 降序
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_media_type_filter(memory_store, fake_embedding):
    _seed_store(memory_store, fake_embedding)
    engine = SearchEngine(fake_embedding, memory_store)
    results = engine.text_search("hello", top_k=10, media_type="image")
    assert all(r.media_type == "image" for r in results)


def test_video_frame_dedup(memory_store, fake_embedding):
    _seed_store(memory_store, fake_embedding)
    engine = SearchEngine(fake_embedding, memory_store)
    results = engine.text_search("hello", top_k=10, media_type="video_frame")
    # 同一视频只应出现一次
    paths = [r.file_path for r in results]
    assert paths.count("/videos/v.mp4") == 1


def test_hybrid_dedup_videos(memory_store, fake_embedding):
    _seed_store(memory_store, fake_embedding)
    engine = SearchEngine(fake_embedding, memory_store)
    results = engine.hybrid_search("hello", top_k=10)
    vid_paths = [r.file_path for r in results if r.media_type == "video_frame"]
    assert len(vid_paths) == len(set(vid_paths))  # 视频帧无重复视频
