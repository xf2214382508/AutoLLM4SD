import os
import pickle
import networkx as nx
import numpy as np
from openai import OpenAI
#import torch


# 加载图数据
def load_graph():
    graph_path = "dataset/Karate/Graph1.pkl" 
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

# 加载训练数据
def load_train_data():
    dataset_path = "dataset/Karate/processed_dataset/train_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)
    return train_dataset

# 创建OpenAI客户端
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 加载数据
G = load_graph()
train_dataset = load_train_data()

# 获取图的边信息
edges_list = list(G.edges())
edges_info = ", ".join([f"({u}, {v})" for u, v in edges_list])

# 获取第一个样本（可以修改索引以使用不同样本）
sample_idx = 0
sample = train_dataset[sample_idx]
node_features = sample['node_features']  # shape: (time_steps, N, hidden_size) 或文本列表
print(node_features.shape)
source_nodes = sample['source_nodes']  # 源节点列表
print(source_nodes)


# 基本变量
propagation_model = "SIR model"
case_examples = f"Case examples from historical data, e.g., nodes {source_nodes} are known propagation sources"

# 创建每个时间步的提示
time_steps = node_features.shape[0] if isinstance(node_features, np.ndarray) else len(node_features)
completions = []  # 存储所有时间步的completion结果

# 为每个时间步分别调用API
for t in range(time_steps):
    # 获取当前时间步的节点特征
    if isinstance(node_features, np.ndarray):
        current_features = node_features[t]
        nodes_info = ", ".join([f"node{i}: {list(current_features[i])}" for i in range(len(current_features))])
    else:
        current_features = node_features[t]
        nodes_info = ", ".join([f"node{i}: {current_features[i]}" for i in range(len(current_features))])
    
    # 创建当前时间步的提示
    current_prompt = f"""You are an expert in propagation dynamics. The current propagation model is {{{propagation_model}}}. Given a graph with nodes and edges, where the nodes at timestep {t} are: {{{nodes_info}}}, and the edges are: {{edge_list: {edges_info}}}.
Your task is to predict which nodes are propagation sources. For each node, select one option from {{2 options [non-propagation source, propagation source]}} that you think is most likely. You need to make predictions based on the dynamics and rules of the propagation model and empirical cases {{{case_examples}}}.
Please output your final answer in the format: Answer: {{node1: non-propagation source, node2: propagation source, ...}}"""
    
    # 为当前时间步创建独立的消息列表
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in network propagation dynamics."},
        {"role": "user", "content": current_prompt}
    ]
    
    # 调用API获取当前时间步的completion
    print(f"正在处理时间步 {t}...")
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=messages,
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    
    # 存储completion结果
    completions.append(completion)
    print(f"时间步 {t} 处理完成")

# 打印所有completion结果
print(f"\n总共获得 {len(completions)} 个completion结果:")
for i, completion in enumerate(completions):
    print(f"\n=== 时间步 {i} 的结果 ===")
    print(completion.model_dump_json())