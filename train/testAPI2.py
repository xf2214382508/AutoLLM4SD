# pip install openai
from openai import OpenAI
import numpy as np
import os

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========= Step 1: 定义推理问题 =========
question = """
有三个盒子：
A: 两个红球
B: 两个蓝球
C: 一个红球一个蓝球。
随机选择一个盒子，再随机取一个球，发现它是红的。
问：这个球来自盒子A（全红盒子）的概率是多少？
"""

# ========= Step 2: 生成多个思维分支 =========
branch_prompt = f"""
你是一个逻辑推理专家。请针对以下问题，生成多种不同的推理思路。
问题：{question}

要求：
1. 给出至少3种不同的推理路径。
2. 每个推理路径分为若干步（Step 1, Step 2, ...），并推导出一个最终答案。
3. 保留中间推理过程，不要只写答案。
"""

branches = client.chat.completions.create(
    model="qwen-plus",  # 或 "gpt-5"（如果有权限）
    messages=[{"role": "user", "content": branch_prompt}],
    n=3,  # 生成3份不同推理分支
    temperature=0.8,
)

thoughts = [c.message.content for c in branches.choices]

print("=== 🧩 生成的思维分支 ===")
for i, t in enumerate(thoughts):
    print(f"\n--- 分支 {i+1} ---\n{t}\n")


# ========= Step 3: 模型评估每个推理分支 =========
scores = []
for i, t in enumerate(thoughts):
    eval_prompt = f"""
请评价以下推理路径的合理性与逻辑一致性。
评分范围：0到10。
问题：{question}
推理路径：
{t}
请输出格式为：Score = [分数]，并说明理由。
"""
    evaluation = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.3,
    )
    text = evaluation.choices[0].message.content
    print(f"\n=== 🔍 分支 {i+1} 评估 ===\n{text}\n")

    # 简单提取数字评分
    try:
        score = float(text.split("Score")[1].split("=")[1].split()[0])
    except Exception:
        score = 5.0
    scores.append(score)

# ========= Step 4: 选择最优路径并总结 =========
best_index = int(np.argmax(scores))
best_thought = thoughts[best_index]

final_prompt = f"""
以下是最优推理路径：
{best_thought}

请基于该路径，总结最终结论，并用一句话解释答案。
"""

final_answer = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": final_prompt}],
    temperature=0.2,
)

print("\n=== 🧠 最终推理结果 ===")
print(final_answer.choices[0].message.content)
