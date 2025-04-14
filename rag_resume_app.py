import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
from resume_docs import docs
import PyPDF2  # 用于PDF读取
import os
import re

# 初始化大模型客户端（DeepSeek）
client = OpenAI(
    api_key="a104bfaefd8152c7ce92eabf0b576cefaad307cd",
    base_url="https://api-w1ke45ednco9tce8.aistudio-app.com/v1"
)

# 编码知识库
st.write("正在初始化向量数据库……")
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))
st.write("向量数据库初始化完成！")


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

# 从PDF或TXT中提取文字
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        return "不支持的文件类型，请上传 PDF 或 TXT 文件。"


# 分句
def build_faiss_index(text):
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
    embeddings = model.encode(chunks)

    # 构建向量索引
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks, embeddings


# 上下文记忆功能（多轮对话）
chat_history = []
def generate_answer_with_memory(query, context, history):
    messages = [
        {"role": "system", "content": "你是一个专业的简历优化助手，请根据上下文和用户问题进行回答。"}
    ]

    # 添加历史聊天记录
    messages.extend(history)

    # 当前轮问题 + 简历上下文
    messages.append({"role": "user", "content": f"简历内容如下：\n{context}\n\n问题：{query}"})

    # 请求大模型
    response = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        temperature=0.6,
        messages=messages
    )

    # 解析回答
    answer = response.choices[0].message.content

    # 把这轮问答加入历史
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history


# Streamlit 页面
st.title("📄 简历优化问答系统")
# st.write("请输入你的简历优化相关问题，我将为你检索资料并生成专业建议~")
st.write("请上传你的简历，我将为你检索资料并生成专业建议~")

# 上传简历文档
uploaded_file = st.file_uploader("上传简历文件（支持 PDF 或 TXT）", type=["pdf", "txt"])
resume_text = ""

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'clear_chat_flag' not in st.session_state:
    st.session_state.clear_chat_flag = False

# 简历上传成功
if uploaded_file:
    resume_text = extract_text_from_file(uploaded_file)
    st.success("简历上传成功！以下是简历内容预览：")
    st.text_area("简历内容", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=300)

qa_placeholder = st.empty()
with qa_placeholder.container():
    if st.session_state.get("clear_chat_flag", False):
        st.session_state.query = ""  # ⬅️ 在输入框渲染前清空
        st.session_state.clear_chat_flag = False
        st.stop()

    # 问答
    if uploaded_file and resume_text:
        index, chunks, embeddings = build_faiss_index(resume_text)

        query = st.text_input("💬 请提问：", key="query", placeholder="比如：这份简历有什么可以优化的地方？")

        if query:
            context = retrieve_context(query, chunks, embeddings, index)
            answer, st.session_state.chat_history = generate_answer_with_memory(query, context, st.session_state.chat_history)
            thought, clean_answer = extract_thought_and_answer(answer)

            st.markdown("### 📚 检索到的上下文：")
            st.info(context)

            st.markdown("### 🤔 思考")
            st.info(thought if thought else "无")
            st.markdown("### 🤖 回答：")
            st.success(clean_answer)

            st.markdown("### 🗂️ 聊天记录：")
            for msg in st.session_state.chat_history:
                role = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"**{role}：** {msg['content']}")

# 清除聊天记录
st.divider()
if st.button("🧼 清除聊天记录"):
    st.session_state.chat_history = []
    st.session_state.clear_chat_flag = True
    qa_placeholder.empty()
    st.rerun()