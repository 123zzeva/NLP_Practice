import time
import json
import psutil
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import logging

# 假设以下函数已从系统模块中导入
from rag_resume_app import retrieve_context, generate_answer_with_memory, build_faiss_index, extract_thought_and_answer
from resume_docs import docs

# ----------------------- 基础初始化 -----------------------
# 加载 SentenceTransformer 模型，用于检索时的文本编码
model = SentenceTransformer("all-MiniLM-L6-v2")

# 构建知识库索引
doc_embeddings = model.encode(docs)
retrieval_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
retrieval_index.add(np.array(doc_embeddings))
logging.info("知识库向量索引初始化完成")

# ----------------------- 测试数据准备 -----------------------
# 示例测试集（查询及期望返回的上下文片段参考，可以依据实际数据扩充）
TEST_DATA = [
    {
        "query": "简历中有哪些地方可以突出项目经历？",
        "expected_context": "项目经历是关键，要详细描述项目背景、目标、自己承担的角色及具体工作内容"
    },
    {
        "query": "如何描述实习经历以便更有说服力？",
        "expected_context": "实习经历采用STAR法则：情境(Situation)→任务(Task)→行动(Action)→结果(Result)"
    },
    # 添加更多测试用例……
]


# # ----------------------- 检索效果评估 -----------------------
# def evaluate_retrieval(test_data, top_k=5):
#     """
#     计算检索模块的精准率、召回率、F1 值以及 Top-k 准确率。
#     这里假设 expected_context 字段中包含了关键的词或短语
#     """
#     total_queries = len(test_data)
#     hits_topk = 0
#     precision_sum = 0.0
#     recall_sum = 0.0
#     for item in test_data:
#         query = item["query"]
#         expected = item["expected_context"]
#         # 构造待检索文本，将简历全文分句（此处简单地以换行符划分，实际可以使用更精细的分句处理）
#         # 这里假设 "resume_text" 为整个简历文本，
#         # 评估时可用模拟数据，例如直接使用知识库中的内容
#         resume_text = "\n".join(docs)
#         # 构建向量索引和切分块
#         index, chunks, embeddings = build_faiss_index(resume_text)
#         retrieved_context = retrieve_context(query, chunks, embeddings, index, top_k=top_k)
#
#         # 判断返回的 Top-k 中是否包含预期的关键短语（可以改为更加复杂的匹配）
#         if expected in retrieved_context:
#             hits_topk += 1
#
#         # 此处简单地将匹配作为一个二分类问题，精准率和召回率均为1/0（实际可以利用标签匹配多个关键词）
#         precision_sum += 1.0 if expected in retrieved_context else 0.0
#         recall_sum += 1.0 if expected in retrieved_context else 0.0
#
#     topk_accuracy = hits_topk / total_queries
#     precision = precision_sum / total_queries
#     recall = recall_sum / total_queries
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#
#     print("【检索效果评估】")
#     print("Top-{}准确率：{:.2f}".format(top_k, topk_accuracy))
#     print("精准率：{:.2f}".format(precision))
#     print("召回率：{:.2f}".format(recall))
#     print("F1 分数：{:.2f}".format(f1))
#     return {
#         "topk_accuracy": topk_accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }
#

# ----------------------- 回答质量评估 -----------------------
def evaluate_answer_quality(test_data):
    """
    对生成回答与期望答案进行 ROUGE 和 BLEU 评估，
    注意：这里期望答案需要提前定义，此处用 expected_response 模拟
    """
    rouge_evaluator = Rouge()
    bleu_scores = []
    rouge_scores = []

    total_response_time = 0.0

    # 对于多轮对话可以维护一个历史记录，这里简单采用空历史
    chat_history = []

    for item in test_data:
        query = item["query"]
        # 这里期望答案仅供参考，对应于人工标注的回答文本
        expected_response = item.get("expected_response", item["expected_context"])  # 作为示例

        resume_text = "\n".join(docs)  # 模拟“简历全文”
        index, chunks, embeddings = build_faiss_index(resume_text)
        retrieved_context = retrieve_context(query, chunks, embeddings, index, top_k=5)

        start_time = time.time()
        generated_response, chat_history = generate_answer_with_memory(query, retrieved_context, chat_history)
        elapsed = time.time() - start_time
        total_response_time += elapsed

        # 如果返回内容中含有思考部分，可分离得到真正回答
        _, clean_answer = extract_thought_and_answer(generated_response)

        # 使用 ROUGE 进行对比
        rouge_score = rouge_evaluator.get_scores(clean_answer, expected_response)[0]['rouge-l']['f']
        rouge_scores.append(rouge_score)

        # 使用 BLEU 进行对比（tokenize 可根据需求自行调整）
        reference = expected_response.split()
        candidate = clean_answer.split()
        bleu = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu)

        print("Query: ", query)
        print("生成回答: ", clean_answer)
        print("ROUGE-L: {:.2f}, BLEU: {:.2f}, 响应时间: {:.2f}s".format(rouge_score, bleu, elapsed))
        print("—————————————————————\n")

    avg_response_time = total_response_time / len(test_data)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print("【回答质量评估】")
    print("平均响应时间：{:.2f}s".format(avg_response_time))
    print("平均 ROUGE-L 分数：{:.2f}".format(avg_rouge))
    print("平均 BLEU 分数：{:.2f}".format(avg_bleu))

    return {
        "avg_response_time": avg_response_time,
        "avg_rouge": avg_rouge,
        "avg_bleu": avg_bleu,
    }


# ----------------------- 资源消耗评估 -----------------------
def evaluate_resource_consumption(num_iterations=5, query="测试查询"):
    """
    模拟多次请求，统计 CPU 和内存的平均消耗
    """
    process = psutil.Process()
    cpu_percentages = []
    memory_usages = []

    # 使用相同的上下文，构造简单的评估场景（实际应涵盖完整请求流程）
    resume_text = "\n".join(docs)
    index, chunks, embeddings = build_faiss_index(resume_text)
    retrieved_context = retrieve_context(query, chunks, embeddings, index, top_k=5)
    chat_history = []

    for i in range(num_iterations):
        # 记录请求前 CPU 与内存占用
        before_cpu = psutil.cpu_percent(interval=None)
        before_mem = process.memory_info().rss / (1024 * 1024)  # MB

        start_time = time.time()
        # 调用生成回答接口
        _, chat_history = generate_answer_with_memory(query, retrieved_context, chat_history)
        elapsed = time.time() - start_time

        after_cpu = psutil.cpu_percent(interval=None)
        after_mem = process.memory_info().rss / (1024 * 1024)  # MB

        cpu_percentages.append(after_cpu)
        memory_usages.append(after_mem)
        print("Iteration {}: 时间 {:.2f}s, CPU {:.2f}%, 内存 {:.2f}MB".format(i + 1, elapsed, after_cpu, after_mem))

    avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
    avg_mem = sum(memory_usages) / len(memory_usages)

    print("【资源消耗评估】")
    print("平均 CPU 使用率：{:.2f}%".format(avg_cpu))
    print("平均内存消耗：{:.2f}MB".format(avg_mem))

    return {
        "avg_cpu": avg_cpu,
        "avg_mem": avg_mem,
    }


# ----------------------- 多轮对话评估 -----------------------
def evaluate_multi_round_dialogue(test_data, rounds=3):
    """
    模拟多轮对话场景，检查系统是否能在连续对话中保持正确的上下文记忆与关联性
    """
    chat_history = []
    for item in test_data:
        query = item["query"]
        resume_text = "\n".join(docs)
        index, chunks, embeddings = build_faiss_index(resume_text)
        retrieved_context = retrieve_context(query, chunks, embeddings, index, top_k=5)
        print("------ 新对话开始 ------")
        for round_idx in range(rounds):
            # 在每一轮中可能调整问题，例如逐步深入问题
            round_query = query + "，补充问题 {}".format(round_idx + 1)
            response, chat_history = generate_answer_with_memory(round_query, retrieved_context, chat_history)
            _, clean_answer = extract_thought_and_answer(response)
            print("轮次 {}: 问题：{} \n回答: {}\n".format(round_idx + 1, round_query, clean_answer))
        print("------ 对话结束 ------\n")
    # 多轮对话的评测通常依赖于人工对连续性和上下文关联性的评分，此处仅示例输出对话内容


# ----------------------- 主测试流程 -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("开始自动化评估……\n")

    # # 1. 检索效果评估
    # retrieval_metrics = evaluate_retrieval(TEST_DATA, top_k=5)

    # 2. 回答质量评估
    answer_quality_metrics = evaluate_answer_quality(TEST_DATA)

    # 3. 资源消耗评估
    resource_metrics = evaluate_resource_consumption(num_iterations=5, query="测试查询")

    # 4. 多轮对话评估
    evaluate_multi_round_dialogue(TEST_DATA, rounds=3)

    # 将各项结果汇总后保存到 JSON 文件（可选）
    eval_results = {
        # "retrieval": retrieval_metrics,
        "answer_quality": answer_quality_metrics,
        "resource": resource_metrics,
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)

    print("\n自动化评估完成，结果已保存到 evaluation_results.json")