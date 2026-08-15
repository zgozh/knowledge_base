# 掌柜智库 · 知识库系统（Knowledge Base）

面向**商品手册 / 产品文档**的领域 RAG（检索增强生成）知识库。基于 LangGraph 编排两条流水线——「导入」与「查询」，实现从 PDF/Markdown 文档的解析、切片、向量化、入库，到多路检索、融合重排、问答生成的完整闭环。

---

## 一、功能特性

- **文档导入**：PDF（MinerU 结构化解析）/ Markdown 双格式入参
- **图片理解**：抽取文档内图片，调用多模态大模型生成摘要，上传 MinIO 并替换引用
- **商品主体识别**：LLM 自动识别文档中的商品/产品名称，作为检索过滤元数据
- **混合向量检索**：BGE-M3 稠密 + 稀疏双向量，Milvus 混合检索
- **多路召回 + 融合**：向量检索 / HyDE（假设性答案检索）/ 联网搜索（MCP）三路并行，RRF 融合
- **精排重排**：qwen3-rerank 交叉编码器打分 + 断崖检测动态截断 Top-K
- **流式问答**：SSE 流式输出，会话历史落 MongoDB
- **异步导入**：后台任务 + 任务进度轮询
- **效果评测**：RAGAS 评测脚本

---

## 二、架构总览

### 导入流水线（`processor/import_processor`）

```
node_entry
   ├─ .pdf ──> node_pdf_to_md (MinerU API)
   └─ .md  ──> ──────────────────┐
                                 ▼
                          node_md_img (图片摘要 + MinIO)
                                 ▼
                        node_document_split (切片)
                                 ▼
                   node_item_name_recognition (商品名识别)
                                 ▼
                       node_bge_embedding (BGE-M3 向量化)
                                 ▼
                       node_import_milvus (入库)
```

### 查询流水线（`processor/query_processor`）

```
node_item_name_confirm (商品型号确认)
   ├─ 已能直接回答 / 需反问 / 查无 ──> node_answer_output
   └─ 需要检索 ──> node_multi_search ─┬─> node_search_embedding (向量检索)
                                      ├─> node_search_embedding_hyde (HyDE)
                                      └─> node_web_search_mcp (联网搜索)
                                            ▼
                                       node_join (合并)
                                            ▼
                                       node_rrf (融合排序)
                                            ▼
                                       node_rerank (精排 + 断崖截断)
                                            ▼
                                       node_answer_output (生成)
```

### 存储与模型

| 层 | 技术 | 用途 |
|---|---|---|
| 编排 | LangGraph | 两条流水线状态图 |
| 向量库 | Milvus | 稠密+稀疏混合检索（`kb_chunks` / `kb_item_names`） |
| 文档库 | MongoDB | 会话历史记录 |
| 对象存储 | MinIO | 原始文件 + 图片 |
| 嵌入 | BGE-M3（本地） | 稠密 + 稀疏向量 |
| 重排 | qwen3-rerank（DashScope API） | 交叉编码精排 |
| 大模型 | Qwen 系列（DashScope API） | 文本生成 / 多模态 / 商品识别 |
| 文档解析 | MinerU（云 API） | PDF → Markdown |
| 评测 | RAGAS | 检索/生成质量评估 |

---

## 三、技术栈

- **语言**：Python 3.12+
- **包管理**：`uv`（依赖锁定在 `uv.lock`）
- **Web**：FastAPI + Uvicorn（SSE 流式）
- **编排**：LangGraph、LangChain
- **检索**：Milvus（`pymilvus[model]`）、BGE-M3（FlagEmbedding/transformers）
- **模型**：transformers、torch（CUDA）、dashscope、modelscope
- **存储**：pymongo、minio
- **其他**：python-dotenv、colorlog

---

## 四、目录结构

```
knowledge_base/
├── config/                 # 配置单例（dataclass + 环境变量）
│   ├── embedding_config.py # BGE-M3 嵌入配置
│   ├── lm_config.py        # LLM 配置
│   ├── milvus_config.py    # Milvus 配置
│   ├── mineru_config.py    # MinerU 配置
│   ├── minio_config.py     # MinIO 配置
│   ├── reranker_config.py  # 重排配置
│   └── bailian_mcp_config.py # 联网搜索 MCP 配置
├── processor/
│   ├── import_processor/   # 导入流水线（节点 + 状态 + 主图）
│   │   ├── main_graph.py   #   导入图定义
│   │   ├── base.py         #   节点基类（日志/异常/任务追踪）
│   │   ├── state.py        #   图状态
│   │   └── nodes/          #   node_entry / pdf_to_md / md_img / split / item_name / bge / milvus
│   └── query_processor/    # 查询流水线（节点 + 状态 + 主图）
│       ├── main_graph.py   #   查询图定义（main_graphV2.py 为演进版本）
│       ├── base.py         #   节点基类
│       ├── state.py        #   图状态
│       └── nodes/          #   item_name_confirm / search / hyde / web_search / rrf / rerank / answer
├── utils/                  # 公共工具（embedding / milvus / mongo / minio / llm / rerank / sse / task）
├── web/                    # FastAPI 服务层 + 前端页面
│   ├── api/
│   │   ├── import_service.py # 导入 API（端口 8000）
│   │   └── query_service.py  # 查询 API（端口 8001）
│   └── page/
│       ├── import.html     # 上传页
│       └── chat.html       # 对话页
├── eval/                   # RAGAS 评测（evaluate_ragas.py + 问答集）
├── test/                   # 学习/验证用的零散脚本（非单元测试套件）
├── main.py                 # 模型下载脚本（bge-reranker-large）
├── .env.example            # 环境变量模板
└── pyproject.toml          # 依赖与 uv 源配置
```

---

## 五、环境要求

### 硬件 / 软件

- **Python** 3.12+
- **`uv`** 包管理器（[安装](https://docs.astral.sh/uv/)）
- **NVIDIA GPU + CUDA**（BGE-M3 本地嵌入默认跑在 `cuda:0`；无 GPU 可改 `.env` 中 `BGE_DEVICE=cpu`，但会慢很多）

### 外部服务（需先启动）

| 服务 | 默认地址 | 说明 |
|---|---|---|
| Milvus | `http://localhost:19530` | 向量库（可 Docker 启动 `milvusdb/milvus`） |
| MongoDB | `mongodb://localhost:27017` | 会话历史 |
| MinIO | `localhost:9000` | 对象存储（默认 `minioadmin`/`minioadmin`） |

### 需要申请的密钥

| 密钥 | 用途 |
|---|---|
| `MINERU_API_TOKEN` | MinerU 云 API，PDF 解析（https://mineru.net） |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | 阿里云 DashScope（LLM / VL / rerank / 联网搜索） |

> 只导入 Markdown 文件、不做 PDF 解析时，可暂不配置 MinerU。

---

## 六、快速开始

### 1. 克隆并安装依赖

```bash
git clone <repo-url>
cd knowledge_base
uv sync
```

> `pyproject.toml` 已配置 `torch` 从 NVIDIA cu128 源安装，`uv sync` 会自动处理。

### 2. 配置环境变量

```bash
Copy-Item .env.example .env   # PowerShell
# 或 Linux/macOS: cp .env.example .env
```

编辑 `.env`，至少填好：
- `OPENAI_API_KEY` / `DASHSCOPE_API_KEY`（DashScope）
- `MINERU_API_TOKEN`（如需 PDF 解析）
- `BGE_M3_PATH` / `BGE_RERANKER_LARGE`（本地模型路径）
- `DATA_BASED_ROOT_DIR`（上传文件落盘目录）
- Milvus / MongoDB / MinIO 地址（如与默认不同）

### 3. 下载本地模型

BGE-M3（稠密+稀疏嵌入，必需）：

```bash
uv run python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache')"
```

BGE-reranker-large（原 `main.py` 已封装）：

```bash
uv run python main.py
```

> 路径需与 `.env` 中 `BGE_M3_PATH`、`BGE_RERANKER_LARGE` 一致。

### 4. 启动外部服务

确保 Milvus、MongoDB、MinIO 已启动并可连通（可用 Docker Compose 自行编排，项目暂未内置）。

### 5. 启动应用（两个服务分别起）

导入服务（端口 **8000**）：

```bash
uv run python -m web.api.import_service
# 或：uv run uvicorn web.api.import_service:app --host 127.0.0.1 --port 8000
```

查询服务（端口 **8001**）：

```bash
uv run python -m web.api.query_service
# 或：uv run uvicorn web.api.query_service:app --host 127.0.0.1 --port 8001
```

### 6. 访问

| 页面 | 地址 | 说明 |
|---|---|---|
| 导入页 | http://127.0.0.1:8000/import.html | 上传 PDF/Markdown，查看处理进度 |
| 对话页 | http://127.0.0.1:8001/chat.html | 知识库问答 |
| 接口文档 | http://127.0.0.1:8000/docs | 导入服务 Swagger |

---

## 七、使用说明

1. 打开导入页（8000），上传 PDF 或 Markdown 文件。
2. 后台 LangGraph 流水线自动执行：解析 → 图片摘要 → 切片 → 商品名识别 → 向量化 → 入库。
3. 轮询 `/status/{task_id}` 查看进度，`completed` 表示入库完成。
4. 打开对话页（8001），输入商品相关问题，系统完成召回 → 融合 → 重排 → 生成并流式返回。

---

## 八、API 接口

### 导入服务（8000）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 多文件上传（form-data），返回 `task_ids` |
| GET | `/status/{task_id}` | 查询任务进度（status / done_list / running_list） |
| GET | `/import.html` | 上传页 |

### 查询服务（8001）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 问答（`query` / `session_id` / `is_stream`） |
| GET | `/stream/{session_id}` | SSE 流式结果 |
| GET | `/history/{session_id}` | 查询会话历史 |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| GET | `/health` | 健康检查 |

---

## 九、评测（RAGAS）

```bash
cd eval
uv run python evaluate_ragas.py
```

- `qa.csv`：评测问答集
- `qa_result.csv`：评测输出（由脚本生成，已 gitignore）
- 评测依赖 `.env` 中的 LLM / Embedding 配置

---

## 十、注意事项 / 已知问题

- **密钥安全**：`.env`、`eval/.env` 已加入 `.gitignore`，切勿提交真实密钥。
- **`web/` 目录尚未纳入版本库**：生产层代码（两个 FastAPI 服务 + 前端）当前未 `git add`，建议纳入版本控制。
- **重复定义**：`utils/milvus_utils.py` 中 `create_hybrid_search_requests` 定义了两次，属遗留重复。
- **图版本并存**：`query_processor` 下 `main_graph.py` 与 `main_graphV2.py` 并存，需明确以哪个为准。
- **重排实现**：当前主图走 DashScope `qwen3-rerank` API；`main.py` 下载的本地 `bge-reranker-large` 与 `.env` 中 `BGE_RERANKER_*` 配置目前未接入主流程。
