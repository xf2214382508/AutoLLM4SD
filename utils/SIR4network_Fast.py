import networkx as nx
import numpy as np
import random
import pickle
#import csv
from scipy.sparse.linalg import eigs
#import h5py
import time # 用于计时
import pandas as pd # 用于高效写入CSV

# 模拟SIR模型的向量化版本，速度极快
def simulate_sir_vectorized(num_nodes, neighbors_dict, initial_state_array, beta, gamma, steps):
    """
    使用Numpy向量化操作高速模拟SIR模型。

    Args:
        num_nodes (int): 节点总数。
        neighbors_dict (dict): 预计算的邻居字典 {node_idx: [neighbor_indices]}。
        initial_state_array (np.ndarray): 初始状态的Numpy数组。
        beta (float): 感染率。
        gamma (float): 康复率。
        steps (int): 模拟步数。

    Returns:
        np.ndarray: (steps, num_nodes) 维度的快照历史记录。
    """
    state_array = initial_state_array.copy()
    snapshot_history = np.zeros((steps, num_nodes), dtype='i1')

    for step in range(steps):
        snapshot_history[step] = state_array

        # --- 关键性能点：向量化状态转移 ---
        # 1. 找出所有易感(0)和感染(1)节点的索引
        susceptible_indices = np.where(state_array == 0)[0]
        infected_indices = np.where(state_array == 1)[0]
        
        if len(infected_indices) == 0: # 如果没有感染者，模拟提前结束
            # 将剩余的快照填充为当前状态
            snapshot_history[step:] = state_array
            break

        # 2. 感染过程 (S -> I)
        # 找出所有与感染者相邻的、且本身是易感的节点
        potential_targets = []
        for i_node in infected_indices:
            potential_targets.extend(neighbors_dict[i_node])
        
        # 去重并只保留仍然易感的节点
        unique_targets = np.unique(potential_targets)
        # 使用集合交集操作找出真正的目标
        s_mask = np.isin(unique_targets, susceptible_indices, assume_unique=True)
        final_targets = unique_targets[s_mask]
        
        # 对所有可能被感染的目标节点，进行一次统一的随机判定
        infection_rand_nums = np.random.rand(len(final_targets))
        newly_infected = final_targets[infection_rand_nums < beta]

        # 3. 康复过程 (I -> R)
        # 对所有已感染节点，进行一次统一的随机判定
        recovery_rand_nums = np.random.rand(len(infected_indices))
        newly_recovered = infected_indices[recovery_rand_nums < gamma]

        # 4. 一次性更新所有状态变化
        state_array[newly_infected] = 1  # 更新为感染
        state_array[newly_recovered] = 2 # 更新为康复

    return snapshot_history


def Generate_Snapshot_Optimized_Fast(Graph, Sample_Number, beta, gamma, Sample_steps, hdf5_path, snapshot_info_path):
    """
    结合了向量化模拟和批量IO的最终极速版本。
    """
    start_time = time.time()
    
    # --- 预计算和映射 ---
    node_list = sorted(list(Graph.nodes()))
    num_nodes = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # --- 关键性能点：预计算邻居列表（Numpy友好格式）---
    print("正在预计算邻接表...")
    neighbors_dict = {
        node_to_idx[node]: [node_to_idx[neighbor] for neighbor in Graph.neighbors(node)]
        for node in node_list
    }
    
    # --- I/O优化：准备批量写入snapshot_info ---
    snapshot_info_records = []

    with h5py.File(hdf5_path, 'w') as f:
        print(f"HDF5文件已创建: {hdf5_path}")
        dset_snapshots = f.create_dataset('snapshots', shape=(Sample_Number, Sample_steps, num_nodes), dtype='i1', chunks=True, compression='gzip')
        dset_labels = f.create_dataset('labels', shape=(Sample_Number,), dtype='i4')
        f.create_dataset('node_ids', data=node_list)

        for index in range(Sample_Number):
            # --- 初始化状态数组 ---
            state_array = np.zeros(num_nodes, dtype='i1')
            # 随机选择一个感染源
            source_node_id = random.choice(node_list)
            source_node_idx = node_to_idx[source_node_id]
            state_array[source_node_idx] = 1

            # --- 调用极速版的模拟函数 ---
            sample_snapshots_array = simulate_sir_vectorized(
                num_nodes, neighbors_dict, state_array, beta, gamma, steps=Sample_steps
            )

            # --- 写入HDF5 ---
            dset_snapshots[index, :, :] = sample_snapshots_array
            dset_labels[index] = source_node_id

            # --- I/O优化：暂存状态统计信息，而不是立即写入 ---
            for step in range(Sample_steps):
                snapshot = sample_snapshots_array[step]
                susc = np.count_nonzero(snapshot == 0)
                infec = np.count_nonzero(snapshot == 1)
                recov = np.count_nonzero(snapshot == 2)
                snapshot_info_records.append({
                    'Sample_index': index,
                    'Time_step': step,
                    'Susceptible': susc,
                    'Infected': infec,
                    'Recover': recov
                })

            if (index + 1) % 100 == 0: # 每100个样本打印一次进度
                elapsed = time.time() - start_time
                print(f'已生成 {index + 1}/{Sample_Number} 个样本... 耗时: {elapsed:.2f} 秒')

    # --- I/O优化：使用Pandas一次性将所有统计信息写入CSV，速度飞快 ---
    print("正在将快照统计信息写入CSV文件...")
    df_snapshot_info = pd.DataFrame(snapshot_info_records)
    df_snapshot_info.to_csv(snapshot_info_path, index=False)

    total_time = time.time() - start_time
    print(f'所有 {Sample_Number} 个样本已生成完毕！总耗时: {total_time:.2f} 秒')


if __name__ == "__main__":
    # 加载图数据
    #with open("data_set/ego-facebook/Graph.pkl", 'rb') as f:
    #    graph = pickle.load(f)

    graph = nx.read_gml("dataset/Karate/karate.gml")
    # ... (参数计算部分保持不变) ...
    A = nx.adjacency_matrix(graph)
    A = A.astype(float)
    eigenvalue = eigs(A, k=1, which='LM', return_eigenvectors=False)
    spectral_radius = abs(eigenvalue[0])
    print("谱半径:", spectral_radius)
    R0 = 2.5
    gamma = 0.4
    beta = 10 * R0 * gamma / spectral_radius
    print("感染率:", beta)
    
    # --- 快速验证：先生成一个小的样本集 ---
    # 您可以按需修改这里的数字
    FAST_TEST_SAMPLE_NUMBER = 20000  # 先生成20000样本来快速验证
    Sample_steps = 30
    
    # 定义输出文件路径
    output_hdf5_file = f"data_set/ego-facebook/simulation_results_{FAST_TEST_SAMPLE_NUMBER}.hdf5"
    output_snapshot_info_file = f"data_set/ego-facebook/snapshot_info_{FAST_TEST_SAMPLE_NUMBER}.csv"

    # --- 调用最终极速优化版本 ---
    Generate_Snapshot_Optimized_Fast(
        graph, 
        FAST_TEST_SAMPLE_NUMBER, 
        beta, 
        gamma, 
        Sample_steps, 
        output_hdf5_file,
        output_snapshot_info_file
    )