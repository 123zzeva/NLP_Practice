import json

# 读取整个文件内容
with open("简历数据1~1000.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

# 按照 "### 简历" 分割，跳过第一个空块
entries = [e.strip() for e in raw_data.split("### 简历") if e.strip()]

processed = []
for idx, entry in enumerate(entries):
    # 构建模型输入内容
    src = "请帮我优化以下简历内容：\n" + entry
    tgt = ""  # 暂时为空，如果你有目标输出，可以写入
    processed.append(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False))

# 保存为一行一个 JSON 格式的文件
with open("resumes_processed.jsonl", "w", encoding="utf-8") as f:
    for line in processed:
        f.write(line + "\n")
