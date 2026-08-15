# -*- coding: utf-8 -*-
"""BUG-2: node_import_milvus insert 后未 flush，导致刚导入的数据无法立即被检索到。"""
from config.milvus_config import milvus_config
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus


class FakeMilvusClient:
    def __init__(self):
        self.calls = []

    def insert(self, collection_name, data):
        self.calls.append(("insert", collection_name))
        return {"ids": [1001, 1002]}

    def flush(self, collection_name):
        self.calls.append(("flush", collection_name))


def test_insert_flushes_collection():
    node = NodeImportMilvus()
    fake = FakeMilvusClient()
    chunks = [
        {
            "file_title": "f", "item_name": "i", "content": "c", "part": 0,
            "dense_vector": [0.0] * 1024, "sparse_vector": {0: 1.0},
        }
    ]
    result = node._step_4_insert_data(fake, chunks)

    col = milvus_config.chunks_collection
    assert fake.calls == [("insert", col), ("flush", col)], "必须先 insert 再 flush，且作用于 chunks 集合"
    assert result[0]["chunk_id"] == 1001
