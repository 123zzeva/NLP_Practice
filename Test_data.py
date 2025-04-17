import json
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from resume_docs import docs

# 初始化模型和向量索引
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# 检索函数
def retrieve_context(query, chunks, embeddings, index, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = "\n".join([chunks[i] for i in indices[0]])
    return context, indices[0]

# 加载测试数据集
def load_test_data(file_path):
    test_cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            test_cases.append({
                "query": data["src"],
                "ground_truth": data["tgt"]  # 假设 tgt 是正确答案
            })
    return test_cases

# 计算匹配度
def is_match(retrieved_context, ground_truth):
    # 简单的匹配逻辑：如果 ground_truth 是 retrieved_context 的子串，则认为匹配
    if ground_truth in retrieved_context:
        return True
    # 或者使用正则表达式进行更复杂的匹配
    return bool(re.search(re.escape(ground_truth), retrieved_context))

# 评估函数
def evaluate_retrieval(test_cases, top_k=5):
    total = len(test_cases)
    correct = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    top_k_correct = 0

    for case in test_cases:
        query = case["query"]
        ground_truth = case["ground_truth"]

        # 检索上下文
        retrieved_context, indices = retrieve_context(query, docs, doc_embeddings, index, top_k)

        # 判断是否匹配
        is_correct = is_match(retrieved_context, ground_truth)
        correct += int(is_correct)

        # 计算精准度和召回率
        retrieved_set = set(indices)
        ground_truth_indices = [i for i, doc in enumerate(docs) if ground_truth in doc]
        relevant_retrieved = len(retrieved_set.intersection(ground_truth_indices))
        precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0
        recall = relevant_retrieved / len(ground_truth_indices) if ground_truth_indices else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_total += precision
        recall_total += recall
        f1_total += f1

        # Top-k准确度
        if any(ground_truth in docs[i] for i in indices):
            top_k_correct += 1

    # 计算平均值
    accuracy = correct / total if total > 0 else 0
    avg_precision = precision_total / total if total > 0 else 0
    avg_recall = recall_total / total if total > 0 else 0
    avg_f1 = f1_total / total if total > 0 else 0
    top_k_accuracy = top_k_correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "top_k_accuracy": top_k_accuracy
    }

# 主函数
if __name__ == "__main__":
    # 加载测试数据
    test_cases = load_test_data("resumes_processed2.jsonl")  # 或其他测试文件

    # 评估检索效果
    results = evaluate_retrieval(test_cases, top_k=5)

    # 输出结果
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Top-5 Accuracy: {results['top_k_accuracy']:.4f}")
