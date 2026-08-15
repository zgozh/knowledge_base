"""pytest 基座配置：把项目根目录加入 sys.path、加载 .env、抑制第三方库日志噪音。"""
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载 .env（config 模块也会加载，这里显式加载保证顺序确定）
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 抑制第三方库的 DEBUG/INFO 刷屏
for _name in [
    "pymongo", "openai", "httpx", "httpcore", "urllib3", "minio",
    "langchain", "langchain_core", "asyncio", "pymilvus", "flag_embedding",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)
