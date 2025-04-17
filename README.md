# NLP_Practice
智能简历优化系统

📁 项目根目录
├── 📁 数据预处理（Data Preprocessing）
│   ├── api.py                     # 用于调用大语言模型 API，批量生成模拟问答数据
│   └── 数据预处理.py             # 将原始生成数据清洗、格式化为模型可用的结构（如QA对）

├── 📁 系统主体（System）
│   ├── rag_resume_app.py          # 项目核心程序，前后端一体化的 RAG 简历问答系统
│   └── resume_docs.py             # 简历文档的向量知识库存储模块（支持FAISS检索）

├── 📁 系统评估与测试（Evaluation & Testing）
│   ├── Test_performance.py        # 测试平均响应时间、ROUGE-L、BLEU、CPU使用率、内存消耗
│   ├── Test_data.py               # 评估精准度、召回率、F1 分数、Top-k 准确率等检索效果
│   └── test-1.py                  # 批量生成回答文本，供人工打分评价（相关性、丰富度、表达质量）

├── 📁 数据文件（Data）
│   ├── 简历数据1~1000.txt        # 简历语料库文本数据，用于模型训练或模拟问答输入
│   └── 简历-脱敏.pdf             # 示例上传的简历文档（已脱敏），用于系统演示和测试

├── 📁 附加资料（Others）
│   ├── requirements.txt           # 所需安装的Python依赖库列表
│   ├── 第三组-PPT.pptx           # 项目展示用的汇报PPT文件
│   ├── 使用示例视频-剪辑版.mp4   # 系统功能展示的精简视频版本（适合课堂播放）
│   └── 使用视频-完整版.mp4       # 全流程功能使用视频（含多轮问答与回答分析）
│   └── 流程图.png       # 项目流程图

