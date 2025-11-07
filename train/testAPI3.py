import os
import json
from openai import OpenAI

# ------------------ 配置 ------------------
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ------------------ 用户输入（示例） ------------------
# graph_mapping 可以是文本描述（已经与 token 维度对齐的映射描述）
# 或者直接是数值向量（list of floats），按你的系统输入格式来传入。
graph_mapping = {
    "type": "mapped_graph_text",  # 或 "vector"
    "value": """[Mapped tokens aligned to LLM token dims] 
node1: [I, S_neighbors=2, I_neighbors=2, R_neighbors=1], ...
(这里放你已经完成映射的图信息文本表示)"""
}

# main_prompt 是需要做 embedding 的主体 prompt（纯文本）
main_prompt = """你是一个研究传播动力学的专家，现在的传播模型是{SIR模型}，
给你一个由中心节点和其一阶、二阶邻接节点构成的图{...}，
当前时间步是{t}，你的任务是预测该中心节点在时间步{t-1}的状态，
从{3个可选项[易感态、感染态、恢复态]}中选择一项你认为最有可能的状态。
请将推理过程与最终答案一并输出。"""

# ------------------ 步骤 1：对 main_prompt 生成 embedding ------------------
# 选择 embedding 模型（根据你的平台/需求可调整）
EMBEDDING_MODEL = "text-embedding-3-large"  # 如需更改，请替换为可用 embedding 模型名

emb_resp = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=main_prompt
)

# emb_resp 的结构通常为 {'data': [{'embedding': [...], 'index':0}], ...}
embedding_vector = emb_resp.data[0].embedding

# ------------------ 步骤 2：构造合并的 user message 内容 ------------------
# 为了可读性与后端可解析性，使用 JSON 把 graph mapping 与 embedding 包起来。
# 如果 graph_mapping.value 是向量（list），就直接放入；若是文本就放文本。
combined_payload = {
    "graph_mapping": graph_mapping,        # 你的映射信息（文本或向量）
    "prompt_embedding": embedding_vector,  # 主体 prompt 的 embedding 向量
    "original_prompt_text": main_prompt    # 可选：保留原始 text 以便模型/审查参考
}

# 将 payload 序列化为字符串作为 chat message 的 content
# 注意：embedding 向量可能很长，序列化后消息体会比较大；视后端限制调整存储/传输策略
user_content = json.dumps(combined_payload, ensure_ascii=False)

# ------------------ 步骤 3：调用 chat completion，传入合并后的内容 ------------------
completion = client.chat.completions.create(
    model="qwen-plus",  # 请根据实际可用模型替换
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        # 这里将合并后的 payload 放入单条 user message 中
        {"role": "user", "content": user_content},
    ],
    # 若需其他模型参数（如 temperature、max_tokens），可在这里添加
    # temperature=0.0,
    # max_tokens=512,
)

# 输出返回结果（JSON）
print(completion.model_dump_json())
