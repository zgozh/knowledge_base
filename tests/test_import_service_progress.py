# -*- coding: utf-8 -*-
"""BUG: import_service 进度追踪把 state 字段名当节点名写入 done 列表。

节点进度应仅由 BaseNode.__call__（用正确节点名）记录，run_graph_task 的
stream 循环只是驱动图执行，不应再手动 add_done_task（否则字段名污染进度）。
"""
import web.api.import_service as import_service


def test_run_graph_task_does_not_pollute_done_list(monkeypatch):
    class FakeWorkflow:
        def run(self, state, stream=False):
            # 模拟 stream_mode="values"：每个 event 是完整 state（含一堆字段名）
            yield {"task_id": "t1", "file_dir": ".", "md_content": "x", "chunks": [], "file_title": "f"}
            yield {"task_id": "t1", "file_dir": ".", "md_content": "x", "chunks": [], "file_title": "f", "item_name": "i"}

    done_calls = []

    monkeypatch.setattr(import_service, "KBImportWorkflow", lambda: FakeWorkflow())
    monkeypatch.setattr(import_service, "add_done_task", lambda tid, name: done_calls.append(name))
    monkeypatch.setattr(import_service, "update_task_status", lambda *a, **k: None)

    import_service.run_graph_task("t1", ".", "somefile.md")

    assert done_calls == [], f"run_graph_task 不应手动 add_done_task，实际调用: {done_calls}"
