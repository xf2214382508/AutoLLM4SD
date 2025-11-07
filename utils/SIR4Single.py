import networkx as nx
import numpy as np
import random
import pickle
import csv
from scipy.sparse.linalg import eigs


def record_snapshot_states(snapshot_list, step_list, file_path, total_nodes):
    """
    记录快照中节点的状态信息
    
    Args:
        snapshot_list: 包含多个快照的列表
        step_list: 包含每个快照的时间步列表
        file_path: 保存状态信息的文件路径
        total_nodes: 图中的总节点数
    """
    for i, snapshot in enumerate(snapshot_list):
        Infec = 0
        Recov = 0
        for key, value in snapshot.items():
            if value == 1:
                Infec += 1
            elif value == 2:
                Recov += 1
        Susc = total_nodes - Infec - Recov
                
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Time_step', 'Susceptible', 'Infected', 'Recover']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Time_step': step_list[i], 'Susceptible': Susc, 'Infected': Infec, 'Recover': Recov})


def Generate_Snapshot(Graph, Sample_Number, beta, gamma, Sample_snapshot_num , Snapshot_info_path):
    labels_Dict = {}
    snapshots_Dict = {}
    Sample_number = 0
    index = 0
    min_node = min(Graph.nodes)
    print(f"最小节点编号是：{min_node},{type(min_node)}  ")
    total_nodes = Graph.number_of_nodes()

    while Sample_number < Sample_Number:

        # 生成一个随机感染源                                                                           # TOTest :这里已经解决了索引是0和1的兼容性了？
        random_number = random.randint(int(min_node), Graph.number_of_nodes() - 1 + int(min_node))  #   图索引从0开始是0，[0, Graph.number_of_nodes()-1], 从1开始是1  [1, Graph.number_of_nodes()]

        I0_node = random_number
        initial_infected = {I0_node: 1}
        state = {node: 0 for node in Graph.nodes() if node not in initial_infected}
        state.update(initial_infected)  # dict {0 0 0 0 at the end 1}
        # print(f'node infectious state:  {state}')
        
        final_state, Sample_snapshot_list, step_list = simulate_sir(Graph, state, beta, gamma, steps=30, sample_snapshot_num=Sample_snapshot_num)
        if len(Sample_snapshot_list) != Sample_snapshot_num or len(step_list) != Sample_snapshot_num: 
            continue
        else:
            labels_Dict[index] = I0_node    
            snapshots_Dict[index] = Sample_snapshot_list
            # 记录每个样本 采样快照的节点状态信息
            record_snapshot_states(Sample_snapshot_list, step_list, Snapshot_info_path, total_nodes)
            Sample_number += 1
            index += 1
            
            print(f'Sample_index: {index}')

    print(f'Sample_number:  {Sample_number}')
    return labels_Dict, snapshots_Dict


# 模拟SIR 流行病传播模型
def simulate_sir(G, state, beta, gamma, steps, sample_snapshot_num, Sample_Start_step=10):
    snapshot_list = []
    step_list = []

    sample_Inf_pop = 10   # 最低采样感染人数(I+R)


    for step in range(steps):
        
        snapshot = state.copy()
        if len(snapshot_list) == sample_snapshot_num:
            break
        else:
            if step >= Sample_Start_step:
                if len(snapshot_list) == sample_snapshot_num - 1:
                    # 记录完整的节点状态快照（包含所有节点），便于后续处理
                    snapshot = state.copy()
                    if len(snapshot_list) == sample_snapshot_num - 1:
                        Infec_R_num = 0 
                        for node in G.nodes():
                            if state[node] == 1 or state[node] == 2:  # 节点是感染状态
                                Infec_R_num += 1

                        if Infec_R_num > sample_Inf_pop:
                            snapshot_list.append(snapshot)
                            step_list.append(step)
                else:
                    snapshot_list.append(snapshot)
                    step_list.append(step)


        # 更新节点状态
        new_state = state.copy()
        for node in G.nodes():
            if state[node] == 1:  # 如果节点是感染状态
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:  # 如果邻居是易感状态
                        if np.random.rand() < beta:  # 以beta的概率感染邻居
                            #if new_state[neighbor] != 1:
                            new_state[neighbor] = 1 

                if np.random.rand() < gamma:  # 以gamma的概率康复
                    new_state[node] = 2     

        state = new_state.copy()
          
    return state, snapshot_list, step_list


if __name__ == "__main__":
  
    file_path = "dataset/Karate/karate.gml"    # 接触网络.gml
    graph = nx.read_gml(file_path, label=None)

    # 加载图数据
    #with open("dataset/dolphins/F_Graph.pkl", 'rb') as f:
    #    graph = pickle.load(f)

    A = nx.adjacency_matrix(graph)
    A = A.astype(float)
    # 计算最大特征值（只计算模最大的特征值）
    eigenvalue = eigs(A, k=1, which='LM', return_eigenvectors=False)
    # 谱半径
    spectral_radius = abs(eigenvalue[0])

    print("谱半径:", spectral_radius)

    
    # 初始化SIR模型的参数

    R0 = 2.5     # 基本再生数  
    gamma = 0.2  # 康复率
    beta = 4 * R0 * gamma / spectral_radius  # 4*感染率
    print("感染率:", beta)  # 感染率0.139
    Sample_Number = 2000     # 设置样本数
    labels_Dict, snapshots_Dict = Generate_Snapshot(graph, Sample_Number, beta, gamma , Sample_snapshot_num=3, Snapshot_info_path="dataset/Karate/Snapshot_info1.csv")
    
    
    # 优化：将 labels_Dict 转换为 NumPy 数组并保存
    labels_array = np.array(list(labels_Dict.values()))
    np.save('dataset/Karate/labels1.npy', labels_array)

    # 优化：使用 numpy.savez_compressed 保存 snapshots_Dict 以减少空间占用
    np.savez_compressed('dataset/Karate/snapshots1.npz', **{str(k): v for k, v in snapshots_Dict.items()})

    with open("dataset/Karate/Graph1.pkl", "wb") as f:
        pickle.dump(graph, f)
    

    