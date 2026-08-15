## 生成问题集

请帮根据eval\hak180产品安全手册_new.md ，生成一组问题集（含答案）。
【格式】
1. 要求csv格式
2. 列头两个： question ，ground_truth
3. 编码格式： UTF-8 BOM
【其他要求】
1. 10道题
2. 请以“HAK180 烫金机” 作为问题的开头
【保存位置】
保存位置在eval目录下，文件名为qa.csv


## 生成评估程序

请帮我生成一个ragas的评估程序。
【输入】
1 读取评估程序当前目录下的qa.csv文件，其中 question 列作为问题，ground_truth 列作为答案。
【工具】
1 ragas评估程序中用到的llm 和 embedding ， 请通过调用
utils\llm.py 和 utils\embedding.py 获取
2 rag流程入口query_app.invoke 参考
processor\query_processor\main_graph.py 的__main__函数
【评估指标】
5个ragas指标 Faithfulness, Answer Relevance, Context Precesion, Context Recall, 
Answer Correctness
【输出】
1 输出格式为csv， 保存在评估程序当前目录下 qa_result.csv 文件中
2 输出9列：question， answer， context， ground_truth, faithfulness, answer_relevance, context_precision,
context_recall, answer_correctness
3 csv的编码格式： UTF-8 BOM
【代码要求】
1 要求每个函数有对应的注释说明
2 核心步骤的函数 用  step_1_xx step_2_xx step_3_xx .. 为函数名的前缀
3 核心ragas 函数 要有参数注释说明

pip install ragas