from modelscope import snapshot_download

# 目标缓存目录，和你.env路径对应
model_dir = snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir="D:/ai_models/modelscope_cache",
    revision="master"
)
print("模型下载完成，路径：", model_dir)