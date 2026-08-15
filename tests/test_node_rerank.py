# -*- coding: utf-8 -*-
"""BUG-5/BUG-6: node_rerank 对 None 字段未兜底、异常处理对 list 用 ** 崩溃。"""
import processor.query_processor.nodes.node_rerank as rerank_mod
from processor.query_processor.nodes.node_rerank import NodeRerank


def test_step_1_merge_handles_none_fields():
    node = NodeRerank()
    state = {"rrf_chunks": None, "web_search_docs": None}
    result = node._step_1_merge_multi_source_docs(state)
    assert result == []


def test_step_1_merge_merges_local_and_web():
    node = NodeRerank()
    state = {
        "rrf_chunks": [{"chunk_id": "c1", "title": "t", "content": "正文"}],
        "web_search_docs": [{"url": "https://x", "title": "w", "snippet": "摘要"}],
    }
    result = node._step_1_merge_multi_source_docs(state)
    assert len(result) == 2
    assert result[0]["source"] == "local"
    assert result[1]["source"] == "web"


def test_step_2_rerank_exception_returns_score_none(monkeypatch):
    def boom(query, contents):
        raise RuntimeError("rerank api failed")

    monkeypatch.setattr(rerank_mod, "rerank_documents", boom)

    node = NodeRerank()
    state = {"rewritten_query": "如何调节温度"}
    merged = [{"content": "a", "title": "A"}, {"content": "b", "title": "B"}]
    result = node._step_2_rerank_merged_docs(state, merged)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(doc.get("score") is None for doc in result)
