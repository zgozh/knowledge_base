"""
RAGAS 评估程序

功能说明：
1. 读取本脚本所在目录下的 qa.csv（列：question / ground_truth）
2. 逐条调用知识库 RAG 流程（KBQueryWorkflow）生成答案与检索上下文
3. 使用 RAGAS 的 5 个指标评估：Faithfulness / Answer Relevance / Context Precision / Context Recall / Answer Correctness
4. 将评估结果保存为 qa_result.csv（UTF-8 编码，含 BOM，共 9 列）

运行方式（在项目根目录下执行）：
    python eval/evaluate_ragas.py
"""

import os
import sys

# 将项目根目录加入 sys.path，确保能导入 utils / processor / config 等包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.embedding_utils import generate_embeddings

# 当前脚本所在目录（eval 目录）
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
# 评估专用环境变量文件（仅本程序使用，存放评分 LLM 的 base_url/api_key/model）
EVAL_ENV_PATH = os.path.join(EVAL_DIR, ".env")
# 输入问题集文件
QA_CSV_PATH = os.path.join(EVAL_DIR, "qa.csv")
# 输出评估结果文件
RESULT_CSV_PATH = os.path.join(EVAL_DIR, "qa_result.csv")

# 每个问题对应的 session_id 前缀，便于在监控/日志中区分
SESSION_PREFIX = "eval_ragas"


class ProjectEmbeddings(Embeddings):
    """
    项目 Embeddings 适配器
    包装 utils/embedding_utils.py 中的 generate_embeddings（BGE-M3 混合向量），
    仅使用其 dense 向量部分，使其符合 langchain Embeddings 接口，供 RAGAS 使用。
    """

    def embed_query(self, text: str) -> list[float]:
        """对单个查询文本生成稠密向量

        :param text: str - 待向量化的查询文本
        :return: list[float] - 稠密向量列表
        """
        result = generate_embeddings([text])
        return result["dense"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对文档文本列表批量生成稠密向量

        :param texts: list[str] - 待向量化的文档文本列表
        :return: list[list[float]] - 每个文档对应的稠密向量列表
        """
        result = generate_embeddings(texts)
        return result["dense"]


def step_1_load_questions(qa_csv_path: str = QA_CSV_PATH) -> list[dict]:
    """
    步骤1：读取问题集
    从指定 CSV 中读取 question 与 ground_truth 两列。

    :param qa_csv_path: str - 问题集 CSV 文件路径，默认读取当前目录下的 qa.csv
    :return: list[dict] - 问题列表，每项含 question / ground_truth 两个字段
    """
    df = pd.read_csv(qa_csv_path, encoding="utf-8-sig")
    # 空值兜底，避免后续处理空字符串
    df = df.fillna("")
    return df.to_dict(orient="records")


def _load_eval_env() -> dict:
    """
    加载评估专用 LLM 环境变量
    读取 eval/.env 中的 EVAL_LLM_API_KEY / EVAL_LLM_BASE_URL / EVAL_LLM_MODEL，
    用于构造仅供 RAGAS 评分使用的 LLM 客户端（不影响全局 RAG 流程）。

    :return: dict - 包含 api_key / base_url / model 三个字段
    """
    load_dotenv(EVAL_ENV_PATH)
    return {
        "api_key": os.getenv("EVAL_LLM_API_KEY"),
        "base_url": os.getenv("EVAL_LLM_BASE_URL"),
        "model": os.getenv("EVAL_LLM_MODEL"),
    }


def step_2_build_rag_env() -> tuple:
    """
    步骤2：构建 RAG 评估环境
    初始化 RAG 工作流、RAGAS 所需的 LLM 与 Embeddings 包装器。

    :return: tuple - (workflow, llm_wrapper, embeddings_wrapper)
              - workflow: KBQueryWorkflow 实例，用于生成答案与上下文
              - llm_wrapper: RAGAS 的 LLM 包装器
              - embeddings_wrapper: RAGAS 的 Embeddings 包装器
    """
    # 1. RAG 工作流实例（懒加载，首次调用时编译）
    workflow = KBQueryWorkflow()

    # 2. 构造评分专用 LLM（DeepSeek 官方 API），仅用于 RAGAS 评估
    #    从 eval/.env 读取配置，避免把密钥硬编码在源码中
    eval_llm_cfg = _load_eval_env()
    llm = ChatOpenAI(
        model=eval_llm_cfg["model"],
        api_key=eval_llm_cfg["api_key"],
        base_url=eval_llm_cfg["base_url"],
        temperature=0,
    )
    llm_wrapper = LangchainLLMWrapper(llm)

    # 3. 通过 utils/embedding_utils.py 获取 Embeddings（包装为 RAGAS 可用的接口）
    embeddings = ProjectEmbeddings()
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    return workflow, llm_wrapper, embeddings_wrapper


def step_3_run_rag(workflow: KBQueryWorkflow, question: str, index: int) -> tuple:
    """
    步骤3：调用 RAG 流程生成答案与检索上下文
    执行 KBQueryWorkflow，从最终状态中提取答案与重排序后的文档作为上下文。

    :param workflow: KBQueryWorkflow - 编译后的 RAG 工作流实例
    :param question: str - 用户问题
    :param index: int - 问题序号（用于生成唯一的 session_id）
    :return: tuple - (answer, contexts)
              - answer: str - RAG 生成的最终答案
              - contexts: list[str] - 重排序后文档的 content 列表（作为评估上下文）
    """
    # 1. 构造初始状态（session_id 必须存在，各节点依赖）
    init_state = {
        "session_id": f"{SESSION_PREFIX}_{index}",
        "original_query": question,
        "is_stream": False,  # 关闭流式输出，避免依赖 SSE 推送
    }

    # 2. 执行工作流
    final_state = workflow.run(init_state)

    # 3. 提取答案与上下文
    answer = final_state.get("answer", "")
    reranked_docs = final_state.get("reranked_docs") or []
    contexts = [doc.get("content", "") for doc in reranked_docs if doc.get("content")]

    return answer, contexts


def step_4_build_dataset(records: list[dict], workflow: KBQueryWorkflow) -> EvaluationDataset:
    """
    步骤4：构建 RAGAS 评估数据集
    逐条调用 RAG 流程，将 (问题, 答案, 上下文, 参考答案) 组装为 RAGAS 评估数据集。

    :param records: list[dict] - step_1 读取的问题记录列表
    :param workflow: KBQueryWorkflow - RAG 工作流实例
    :return: EvaluationDataset - RAGAS 评估数据集
    """
    samples = []
    for index, record in enumerate(records):
        question = record["question"]
        ground_truth = record["ground_truth"]

        # 调用 RAG 流程，得到答案与上下文
        answer, contexts = step_3_run_rag(workflow, question, index)

        # 组装 RAGAS 单轮样本
        sample = SingleTurnSample(
            user_input=question,          # 用户问题
            response=answer,              # RAG 生成的答案
            retrieved_contexts=contexts,  # 检索到的上下文文档列表
            reference=ground_truth,       # 参考答案（ground_truth）
        )
        samples.append(sample)
        print(f"[进度] 第 {index + 1}/{len(records)} 题已生成：{question}")

    return EvaluationDataset(samples=samples)


def step_5_run_evaluation(
    dataset: EvaluationDataset,
    llm_wrapper: LangchainLLMWrapper,
    embeddings_wrapper: LangchainEmbeddingsWrapper,
) -> pd.DataFrame:
    """
    步骤5：运行 RAGAS 评估
    使用 5 个 RAGAS 指标对数据集进行评分，返回包含各指标分数的 DataFrame。

    :param dataset: EvaluationDataset - step_4 构建的评估数据集
    :param llm_wrapper: LangchainLLMWrapper - RAGAS LLM 包装器
    :param embeddings_wrapper: LangchainEmbeddingsWrapper - RAGAS Embeddings 包装器
    :return: pd.DataFrame - 每个样本的 5 个指标得分
    """
    # 需要使用的 5 个 RAGAS 指标
    metrics = [
        faithfulness,         # 忠实性：答案是否忠实于检索上下文
        answer_relevancy,     # 答案相关性：答案与问题的相关程度
        context_precision,    # 上下文精确率：检索上下文中的相关比例
        context_recall,       # 上下文召回率：检索上下文覆盖参考答案的程度
        answer_correctness,   # 答案正确性：答案与参考答案的语义一致性
    ]

    # 运行评估（allow_nest_asyncio 兼容已在事件循环内运行的情况）
    result = evaluate(
        dataset=dataset,      # 评估数据集
        metrics=metrics,      # 评估指标列表
        llm=llm_wrapper,      # 评估用 LLM
        embeddings=embeddings_wrapper,  # 评估用 Embeddings
        run_config=RunConfig(),
        allow_nest_asyncio=True,
        show_progress=True,
    )

    # 将评估结果转为 DataFrame（行为样本，列为指标名）
    return result.to_pandas()


def step_6_save_result(df: pd.DataFrame, records: list[dict]) -> str:
    """
    步骤6：合并并保存评估结果
    将问题、答案、上下文、参考答案与各指标得分合并为 9 列，写入 qa_result.csv（UTF-8 带 BOM）。

    :param df: pd.DataFrame - step_5 生成的指标得分表（每行对应一个样本）
    :param records: list[dict] - step_1 读取的原始问题记录（用于还原问题/答案/上下文）
    :return: str - 保存的文件路径
    """
    # 指标列名（RAGAS 列名规范）
    metric_columns = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevance",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "answer_correctness": "answer_correctness",
    }

    # 重命名指标列为目标列名
    df = df.rename(columns=metric_columns)

    # 合并原始字段
    output = pd.DataFrame({
        "question": [r["question"] for r in records],
        "answer": [r.get("_answer", "") for r in records],
        "context": [r.get("_context", "") for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })
    output = pd.concat([output, df.reset_index(drop=True)], axis=1)

    # 保证 9 列顺序与内容完整
    columns = [
        "question", "answer", "context", "ground_truth",
        "faithfulness", "answer_relevance", "context_precision",
        "context_recall", "answer_correctness",
    ]
    output = output[columns]

    # 使用 UTF-8 with BOM 编码写入（Excel 中文兼容）
    output.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[完成] 评估结果已保存至：{RESULT_CSV_PATH}")
    return RESULT_CSV_PATH


def main() -> None:
    """
    主流程：串行执行 step_1 ~ step_6
    """
    # 步骤1：读取问题集
    records = step_1_load_questions()
    print(f"[步骤1] 读取到 {len(records)} 道题目")

    # 步骤2：构建评估环境（RAG 工作流 + LLM + Embeddings）
    workflow, llm_wrapper, embeddings_wrapper = step_2_build_rag_env()

    # 步骤4：逐题调用 RAG 流程，构建评估数据集
    dataset = step_4_build_dataset(records, workflow)

    # 步骤5：运行 RAGAS 评估
    df_scores = step_5_run_evaluation(dataset, llm_wrapper, embeddings_wrapper)

    # 将 RAG 生成的答案/上下文回填到 records，供输出合并
    for idx, sample in enumerate(dataset.samples):
        records[idx]["_answer"] = sample.response
        records[idx]["_context"] = "\n".join(sample.retrieved_contexts)

    # 步骤6：保存结果
    step_6_save_result(df_scores, records)


if __name__ == "__main__":
    main()
