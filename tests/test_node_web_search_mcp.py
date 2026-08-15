# -*- coding: utf-8 -*-
"""BUG-3/BUG-4: node_web_search_mcp 在 MCP 失败/无结果时必须优雅降级，返回空列表而不是抛异常或返回 {}。"""
import processor.query_processor.nodes.node_web_search_mcp as mcp_mod
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp


def test_web_search_mcp_failure_degrades_gracefully_and_marks_done(monkeypatch):
    async def fake_mcp_call(self, query):
        raise RuntimeError("McpError: Session terminated")

    done_calls = []

    def fake_add_done_task(task_id, node_name, is_stream=False):
        done_calls.append((task_id, node_name, is_stream))

    monkeypatch.setattr(NodeWebSearchMcp, "_mcp_call", fake_mcp_call)
    monkeypatch.setattr(mcp_mod, "add_done_task", fake_add_done_task)

    node = NodeWebSearchMcp()
    state = {"rewritten_query": "如何调节温度", "session_id": "s1", "is_stream": False}
    result = node.process(state)

    assert result == {"web_search_docs": []}
    assert done_calls == [("s1", "node_web_search_mcp", False)]


def test_web_search_mcp_empty_query_returns_empty_list():
    node = NodeWebSearchMcp()
    state = {"rewritten_query": "", "session_id": "s2", "is_stream": False}
    result = node.process(state)

    assert result == {"web_search_docs": []}


def test_web_search_mcp_parses_pages_and_filters_empty_snippets(monkeypatch):
    class FakeContent:
        text = ('{"pages": ['
                '{"title": "t1", "url": "https://a", "snippet": "  摘要1  "}, '
                '{"title": "t2", "url": "https://b", "snippet": "   "}]}')

    class FakeResult:
        content = [FakeContent()]

    async def fake_mcp_call(self, query):
        return FakeResult()

    monkeypatch.setattr(NodeWebSearchMcp, "_mcp_call", fake_mcp_call)

    node = NodeWebSearchMcp()
    state = {"rewritten_query": "如何调节温度", "session_id": "s3", "is_stream": False}
    result = node.process(state)

    assert result["web_search_docs"] == [
        {"title": "t1", "url": "https://a", "snippet": "摘要1"},
    ]
