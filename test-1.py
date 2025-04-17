import os
import time
import json
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
import re
from resume_docs import docs

# 初始化大模型客户端（DeepSeek）
client = OpenAI(
    api_key="api_key",
    base_url="api_url"
)

# 编码知识库
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# 从PDF文件中提取文字
def extract_text_from_file(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 根据提问检索
def retrieve_context(query, chunks, embeddings, index, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = "\n".join([chunks[i] for i in indices[0]])
    return context

# 查询 + 调用大模型
def generate_answer(query, context):
    messages = [
        {"role": "system", "content": "你是一个简历优化专家，基于用户上传的简历回答问题。"},
        {"role": "user", "content": f"简历内容如下：\n{context}\n\n问题：{query}"}
    ]
    response = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        temperature=0.6,
        messages=messages
    )
    return response.choices[0].message.content

# 拆分思考部分和正式回答部分
def extract_thought_and_answer(answer_text):
    think_match = re.search(r"<think>(.*?)</think>", answer_text, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""
    clean_answer = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()
    return thought, clean_answer

# 分句
def build_faiss_index(text):
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
    embeddings = model.encode(chunks)

    # 构建向量索引
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks, embeddings

# 处理问题列表并保存答案
def process_questions_and_save_answers(resume_path, questions, output_file="answers.json"):
    resume_text = extract_text_from_file(resume_path)
    index, chunks, embeddings = build_faiss_index(resume_text)

    answers = []

    for question in questions:
        # 检索上下文
        context = retrieve_context(question, chunks, embeddings, index)

        # 计时
        start_time = time.time()
        answer = generate_answer(question, context)
        elapsed_time = time.time() - start_time

        # 拆分思考部分和答案
        thought, clean_answer = extract_thought_and_answer(answer)

        # 保存到答案列表
        answers.append({
            "question": question,
            "answer": clean_answer,
            "thought": thought,
            "elapsed_time": elapsed_time
        })

    # 保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    print(f"回答已保存到 {output_file}")

# 定义问题列表
questions = [
    "我的简历中哪些部分需要改善？",
    "如何使我的技能部分更加突出？",
    "根据我的简历，我适合哪些职位？",
    "HR筛选简历时最看重哪些内容？",
    "HR在招聘过程中最看重的非技术性因素是什么？",
    "作为应届毕业生，如何在简历中展示我的优势？",
    "我的简历是否存在常见的写作错误？",
    "一份好的简历应该包括哪些板块？"
]

# 上传的简历文件路径
resume_path = "张彤阳简历.pdf"  # 请确保简历路径正确

# 调用函数处理问题并保存结果
process_questions_and_save_answers(resume_path, questions)
