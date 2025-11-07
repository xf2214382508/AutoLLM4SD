import os
import pickle
import numpy as np
import networkx as nx
import torch    
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


def load_data(data_Graph, data_label, data_snapshot, data_dir):
    """加载原始数据"""
    print(f'Loading data from: {data_dir}')
    
    # 加载图数据
    graph_path = os.path.join(data_dir, data_Graph)
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)


    # 加载标签数据
    label_path = os.path.join(data_dir, data_label)
    if data_label.endswith('.npy'):
        labels_data = np.load(label_path, allow_pickle=True)
        # 安全地提取标签数据
        if labels_data.shape == () and labels_data.dtype == object:
            labels = labels_data.item()
        else:
            labels = labels_data
    else:  # .pkl
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
    print(f'Labels: {len(labels)} samples')

    # 加载快照数据
    snapshot_path = os.path.join(data_dir, data_snapshot)
    if data_snapshot.endswith('.npz'):
        loaded = np.load(snapshot_path, allow_pickle=True)
        if len(loaded.files) == 1:
            key = loaded.files[0]
            data = loaded[key]
            # 检查数据类型和形状，安全地提取数据
            if data.shape == () and data.dtype == object:
                snapshots = data.item()
            else:
                snapshots = data
        else:
            snapshots = {}
            for k in loaded.files:
                data = loaded[k]
                if data.shape == () and data.dtype == object:
                    snapshots[k] = data.item()
                else:
                    snapshots[k] = data
    else:  # .pkl
        with open(snapshot_path, 'rb') as f:
            snapshots = pickle.load(f)
    print(f'Snapshots: {len(snapshots)} samples')

    return graph, labels, snapshots


class BERTEncoder:
    """BERT文本编码器"""
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def encode_texts(self, texts, max_length=128):
        """
        将文本列表编码为BERT特征向量
        对每个文本元素分别进行编码，生成对应的特征表示
        
        Args:
            texts: 文本列表 [text1, text2, ..., textN]
            max_length: 最大序列长度
        Returns:
            torch.Tensor: shape (N, hidden_size) - N个文本的特征向量
        """
        if not texts:
            return torch.empty(0, 768)  # BERT-base hidden size = 768
            
        # Step 1: 批量tokenize所有文本
        encoded = self.tokenizer(
            texts, 
            padding=True,           # 填充到相同长度
            truncation=True,        # 截断超长文本
            max_length=max_length,
            return_tensors='pt'     # 返回PyTorch张量
        )
        
        # Step 2: 移动到计算设备（GPU/CPU）
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Step 3: 通过BERT模型获取特征
        with torch.no_grad():  # 推理模式，不计算梯度
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS]标记的特征作为整个句子的表示
            features = outputs.last_hidden_state[:, 0, :]  # shape: (N, 768)
            
        return features.cpu()  # 返回CPU上的特征向量


def get_state_name(state):
    """
    将状态数字转换为英文描述
    0: Susceptible (易感态)
    1: Infected (感染态) 
    2: Recovered (恢复态)
    """
    state_map = {0: "Susceptible", 1: "Infected", 2: "Recovered"}
    return state_map.get(state, "Unknown")


def generate_node_feature_text(node, state, neighbor_states, graph):
    """
    为节点生成特征文本描述
    格式: "The state of this node is {state}. About its adjacent nodes: 
           the number of Susceptible nodes is {S_count}, 
           the number of Infected nodes is {I_count}, 
           the number of Recovered nodes is {R_count}."
    """
    # 获取节点状态描述
    state_name = get_state_name(state)
    
    # 统计邻接节点的状态
    neighbors = list(graph.neighbors(node))
    s_count = sum(1 for n in neighbors if neighbor_states.get(n, 0) == 0)
    i_count = sum(1 for n in neighbors if neighbor_states.get(n, 0) == 1)  
    r_count = sum(1 for n in neighbors if neighbor_states.get(n, 0) == 2)
    
    feature_text = (f"The state of this node is {state_name}. "
                   f"About its adjacent nodes: "
                   f"the number of Susceptible nodes is {s_count}, "
                   f"the number of Infected nodes is {i_count}, "
                   f"the number of Recovered nodes is {r_count}.")
    
    return feature_text


def process_data(graph, labels, snapshots, time_steps, use_bert=True):
    """
    处理原始数据，生成训练所需的特征和标签
    
    Sample structure:
    {
        'node_features': Tensor of shape (time_steps, N, hidden_size) - 每个时间步每个节点的BERT特征向量
        'Source_label': Tensor of shape (N) - 传播源标签, N为图graph的节点数
        'sample_id': sample_id - 样本ID
        'snapshots': List of snapshots for reference
    }
    """
    print("=" * 50)
    print("开始处理数据...")
    
    # 获取节点列表并按顺序排序（确保"1"到"34"的顺序）
    all_nodes = list(graph.nodes())
    
    # 根据节点类型进行排序
    if all(isinstance(node, (int, np.integer)) for node in all_nodes):
        # 如果节点是整数类型，直接数值排序
        node_list = sorted(all_nodes)
    elif all(isinstance(node, str) for node in all_nodes):
        # 如果节点是字符串类型，按数值排序
        node_list = sorted(all_nodes, key=lambda x: int(x))
    else:
        # 混合类型，转换为字符串后按数值排序
        node_list = sorted(all_nodes, key=lambda x: int(str(x)))
    
    node_num = len(node_list)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    print(f"图节点数: {node_num}")
    print(f"节点类型: {type(node_list[0]) if node_list else 'None'}")
    print(f"节点列表（排序后）: {node_list[:10]}...")  # 显示前10个节点
    print(f"节点到索引映射示例: {dict(list(node_to_idx.items())[:5])}")  # 显示前5个映射
    print(f"标签数据类型: {type(labels)}")
    if isinstance(labels, np.ndarray):
        print(f"标签数组形状: {labels.shape}")
    elif isinstance(labels, dict):
        print(f"标签字典长度: {len(labels)}")

    # 初始化BERT编码器
    bert_encoder = None
    if use_bert:
        print("初始化BERT编码器...")
        bert_encoder = BERTEncoder()

    # 准备数据集
    dataset = []
    less_steps_sample = 0
    invalid_samples = 0

    sample_ids = list(snapshots.keys())
    print(f"总样本数: {len(sample_ids)}")
    print("-" * 50)
    
    for sample_id in tqdm(sample_ids, desc="处理样本"):
        try:
            # 检查标签数据格式并安全访问
            if isinstance(labels, dict):
                if sample_id not in labels:
                    invalid_samples += 1
                    continue
                source_nodes = labels[sample_id]
            else:
                # labels是numpy数组，使用sample_id作为索引
                if int(sample_id) >= len(labels):
                    invalid_samples += 1
                    continue
                source_nodes = labels[int(sample_id)]
            
            snapshot_list = snapshots[sample_id]
            
            # 确保source_nodes是列表格式（支持多个传播源）
            if not isinstance(source_nodes, (list, tuple)):
                # 如果是numpy数组，转换为列表
                if isinstance(source_nodes, np.ndarray):
                    source_nodes = source_nodes.tolist()
                else:
                    source_nodes = [source_nodes]  # 单个值转换为列表
            
            # 确保snapshot_list是列表格式
            if not isinstance(snapshot_list, list):
                # 如果是numpy数组，转换为列表
                if isinstance(snapshot_list, np.ndarray):
                    snapshot_list = snapshot_list.tolist()
                else:
                    invalid_samples += 1
                    continue

            # 确保有足够时间步
            if len(snapshot_list) < time_steps:
                less_steps_sample += 1
                continue

            # 截取所需的时间步
            snapshot_list = snapshot_list[:time_steps]
            
            # 验证每个快照的格式
            valid_snapshots = True
            for i, snap in enumerate(snapshot_list):
                if not isinstance(snap, dict):
                    print(f"警告: 样本 {sample_id} 时间步 {i} 的快照不是字典格式: {type(snap)}")
                    valid_snapshots = False
                    break
                    
            if not valid_snapshots:
                invalid_samples += 1
                continue

            # 为每个时间步生成节点特征
            if use_bert:
                # 生成BERT特征向量
                node_features_by_time = []
                
                for t, snap in enumerate(snapshot_list):
                    # 为当前时间步的所有节点生成文本描述
                    # 注意：使用排序后的node_list，确保节点顺序为1->2->...->34
                    texts_current_time = []
                    for node in node_list:  # node_list已按1->34顺序排序
                        node_state = snap.get(node, 0)  # 默认状态为0（易感态）
                        feature_text = generate_node_feature_text(node, node_state, snap, graph)
                        texts_current_time.append(feature_text)
                    
                    # 使用BERT编码文本为向量
                    features_tensor = bert_encoder.encode_texts(texts_current_time)  # shape: (N, hidden_size)
                    node_features_by_time.append(features_tensor)
                
                # 堆叠为 (time_steps, N, hidden_size)
                node_features = torch.stack(node_features_by_time, dim=0)
            else:
                # 保持文本格式（用于调试）
                node_features_by_time = []
                for t, snap in enumerate(snapshot_list):
                    node_features_current_time = []
                    for node in node_list:  # node_list已按1->34顺序排序
                        node_state = snap.get(node, 0)
                        feature_text = generate_node_feature_text(node, node_state, snap, graph)
                        node_features_current_time.append(feature_text)
                    node_features_by_time.append(node_features_current_time)
                node_features = node_features_by_time

            # 构建 Source_label: Tensor of shape (N,) 标记传播源节点
            Source_label = torch.zeros((node_num,), dtype=torch.bool)
            
            # 处理列表形式的source_nodes（支持多个传播源）
            for n in source_nodes:
                if n in node_to_idx:
                    Source_label[node_to_idx[n]] = True

            # 构建样本
            sample = {
                'node_features': node_features,          # shape: (time_steps, N, hidden_size) 或文本列表
                'Source_label': Source_label,            # shape: (N,) 源节点标签
                'sample_id': sample_id,                  # 样本ID
                'snapshots': snapshot_list,              # 原始快照数据用于参考
                'source_nodes': source_nodes             # 源节点列表
            }

            dataset.append(sample)

            # 定期清理内存
            if len(dataset) % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            invalid_samples += 1
            continue

    print(f'时间步不足的样本数: {less_steps_sample}')
    print(f'无效样本数: {invalid_samples}')
    print(f'成功处理的样本数: {len(dataset)}')
    
    # 显示一个样本的示例
    if len(dataset) > 0:
        sample_example = dataset[0]
        print(f"\n样本示例 (ID: {sample_example['sample_id']}):")
        print(f"源节点: {sample_example['source_nodes']}")
        if use_bert:
            print(f"特征形状: {sample_example['node_features'].shape}")
        else:
            print(f"时间步数: {len(sample_example['node_features'])}")
            print(f"节点数: {len(sample_example['node_features'][0])}")
            print(f"第一个时间步第一个节点的特征: {sample_example['node_features'][0][0]}")
    
    return dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
    
    # 先分割出训练集
    train_data, temp_data = train_test_split(
        dataset, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # 再从剩余数据中分割出验证集和测试集
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, 
        train_size=val_ratio_adjusted, 
        random_state=random_state
    )
    
    return train_data, val_data, test_data


def save_dataset(train_data, val_data, test_data, output_dir):
    """
    保存处理后的数据集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val_dataset.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(os.path.join(output_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"数据集已保存到 {output_dir} 目录")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")


def main():
    # 需要修改的文件名在 load_data() 、  save_dataset() 、  time_steps 里面
    # 加载原始数据
    print("正在加载原始数据...")
    graph, labels, snapshots = load_data('Graph1.pkl', 'labels1.npy', 'snapshots1.npz',"D:\MyCode\Reason_LLM4SD\Reason_LLM4SD\dataset\Karate" )
    print(f"图节点数: {len(graph.nodes())}")
    print(f"图边数: {len(graph.edges())}")
    print(f"标签数量: {len(labels)}")
    print(f"快照数量: {len(snapshots)}")
    
    # 处理数据
    print("正在处理数据...")
    dataset = process_data(graph, labels, snapshots, time_steps=3, use_bert=True)    # time_steps是 数据集时间步设置
    print(f"处理后的样本数: {len(dataset)}")
    
    # 分割数据集
    print("正在分割数据集...")
    train_data, val_data, test_data = split_dataset(dataset)
    
    # 保存数据集
    print("正在保存数据集...")
    save_dataset(train_data, val_data, test_data, "D:\MyCode\Reason_LLM4SD\Reason_LLM4SD\dataset\Karate\processed_dataset")
    
    print("数据处理完成!")


if __name__ == "__main__":
    main()
