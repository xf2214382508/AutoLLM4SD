import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader,Dataset

from utils.tools import *
from utils.EarlyStopping import *

from utils.parsersv2 import parser
import os

from utils.AutoTimes_Gpt2_promptv3 import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
args = parser.parse_args()
device = torch.device("cuda:1")

seed = 2022
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# 创建数据集类
class InfectionDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            sample = self.data[idx]
            x_enc = sample['x_enc']
            edge_index = sample['edge_index']
            edge_enc = sample['edge_enc']  # 新增边状态编码
            node_pos = sample['node_pos']  # 新增节点位置编码
            infec_source_label = sample['infec_source_label']
            infec_pop = sample['infec_pop']
            sample_id = sample['sample_id']
            
            return x_enc, edge_index, edge_enc, node_pos, infec_source_label, infec_pop, sample_id


def hop_1_metric(G, predicted_source_tensor, true_source_tensor):
    """
    计算 Hop-1 指标
    
    参数：
    G: 接触网络图 (networkx 图)
    predicted_source: 模型预测的传播源节点
    true_source: 真实传播源节点
    
    返回：
        返回正确预测样本数 
    """

    predicted_source = predicted_source_tensor.cpu().numpy()  # batch_size * 1
    true_source = true_source_tensor.cpu().numpy()           # batch_size * 1

    neighbor_list = []
    for i in range(len(predicted_source)):
        neighbors = set(G.neighbors(int(predicted_source[i])))    # DataSet 0-1 Problem,Need Edit，和图的节点索引相关，图的第一个节点是1则需要+1，否则不需要
        neighbors.add(int(predicted_source[i]))                   # DataSet 0-1 Problem,Need Edit，和图的节点索引相关，图的第一个节点是1则需要+1，否则不需要
        neighbor_list.append(neighbors)
    
    correct_sample = 0

    # 如果真实传播源在预测源的邻居或源节点中，则认为该预测为正确
    for i in range(len(neighbor_list)):
        source_index = int(true_source[i])    #+ 1                        # DataSet 0-1 Problem,Need Edit
        if source_index in neighbor_list[i]:
            correct_sample += 1

    return correct_sample


def main():

    data_start_time = time.time()

    #####load data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/dolphins/processed_data')
    
    with open(os.path.join(data_dir, 'train_datasetD5.pkl'), 'rb') as ftrain:
        train_data = pickle.load(ftrain)
    with open(os.path.join(data_dir, 'val_datasetD5.pkl'), 'rb') as fval:
        val_data = pickle.load(fval)
    with open(os.path.join(data_dir, 'test_datasetD5.pkl'), 'rb') as ftest:
        test_data = pickle.load(ftest)
    with open('dataset/dolphins/D_Graph.pkl', 'rb') as file:
        contact_graph = pickle.load(file)

    print(f"加载了 {len(train_data)} 个训练样本")
    print(f"加载了 {len(val_data)} 个验证样本")
    print(f"加载了 {len(test_data)} 个测试样本")
    
    
    # 创建数据加载器
    train_dataset = InfectionDataset(train_data)
    val_dataset = InfectionDataset(val_data)
    test_dataset = InfectionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.b_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.b_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.b_size, num_workers=0)

    data_end_time = time.time()
    print('data loading Finished! Time used: {:.3f}mins.'.format((data_end_time - data_start_time) / 60))

    ## model
    model = Model(args) 
    model = model.to(device)
    # model初始化完成
    

    correct = 0
    test_num = 0
    correct_5 = 0
    correct_Hop_1 = 0

    start_time = time.time()

    model_dict = torch.load("./checkpoint/dolphins_V3_1step_Predict_Expert_EdgeandPos.pt")
    model.load_state_dict(model_dict)
    print('Finished loading checkpoint!')
    # model test
    model.eval()
    with torch.no_grad():
            for step, (x_enc, edge_index, edge_enc, node_pos, infec_source_label, infec_pop, sample_id) in enumerate(test_loader):
                # 转换数据类型
                x_enc = x_enc.to(torch.float32)
                x_enc = torch.flip(x_enc, dims=[1])   # 逆序输入节点特征到模型
                edge_index = edge_index.to(torch.long)
                edge_enc = edge_enc.to(torch.float32)  # 新增边状态编码
                edge_enc = torch.flip(edge_enc, dims=[1])  # 逆序边特征
                node_pos = node_pos.to(torch.float32)  # 新增节点位置编码
                node_pos = torch.flip(node_pos, dims=[1])  # 逆序节点位置
                infec_source_label = infec_source_label.to(torch.float32)
                infec_pop = infec_pop.to(torch.float32)
                infec_pop = torch.flip(infec_pop, dims=[1])   # 逆序的感染人数
                
                # 获取批次大小和时间步数
                B, T, N, _ = x_enc.shape  # [batch_size, time_steps, node_num, 3]
                
                # 将数据移至设备
                x_enc = x_enc.to(device)
                edge_index = edge_index.to(device)
                edge_enc = edge_enc.to(device)  # 新增边状态编码
                node_pos = node_pos.to(device)  # 新增节点位置编码
                infec_source_label = infec_source_label.to(device)
                infec_pop = infec_pop.to(device)
                
                
                prompt = (
                   
                )
                
                # 初始化自回归预测的序列
                x_enc_val = x_enc[:, 0:1, :, :] # 从第一个真实时间步开始
                edge_index_val = edge_index[:, 0:1, :, :]
                # 新增特征的初始化
                edge_enc_val = edge_enc[:, 0:1, :, :]
                node_pos_val = node_pos[:, 0:1, :, :]
                
                # 循环14次以生成14个新的时间步
                for _ in range(0):
                    # 使用当前序列进行预测
                    llm_output_pred, _, _, _, _ = model(
                        x_enc_val,
                        edge_index_val,
                        edge_enc_val,
                        node_pos_val,
                        prompt,
                        graph=contact_graph,
                        is_training=False
                    )
                    
                    # 获取对下一个时间步的预测结果
                    new_x_step = llm_output_pred[:, -1:, :, :].detach()
                    
                    # 将新预测的步拼接到序列中
                    x_enc_val = torch.cat((x_enc_val, new_x_step), dim=1)
                    #print(f"x_enc_val: {x_enc_val.shape}")
                    
                    # 扩展edge_index以匹配新序列的长度 (重复使用第一个时间步的边)
                    edge_index_val = edge_index[:, :x_enc_val.shape[1], :, :]
                    #print(f"edge_index_val: {edge_index_val.shape}")
                
                # 在完整的生成序列上运行一次模型，以获得用于计算损失的最终输出
                llm_output, Infec_Prob_squeezed, _, _, _ = model(
                    x_enc_val,
                    edge_index_val,
                    edge_enc_val,
                    node_pos_val,
                    prompt,
                    graph=contact_graph,
                    is_training=False
                )       
              
                source_pred = Infec_Prob_squeezed[:,-1,:]  # 最后一个时间步的感染概率 [B,node_num]
                Infec_B, Infec_N, Infec_Node_Num = Infec_Prob_squeezed.shape
                source_target = infec_source_label[:,-1,:]  # [B,node_num]


                test_num += source_pred.size(0)
                # 计算准确率
                correct += (source_pred.argmax(dim=1) == source_target.argmax(dim=1)).sum().item()
                
                #  Top-k 指标
                k = 5
                top_k_indices = torch.topk(source_pred, k=k, dim=1).indices  # [batch_size, k]
                
                # 获取目标源头的索引
                target_indices = torch.argmax(source_target, dim=1)  # [batch_size]
                
                # 检查目标索引是否在top-k预测中
                for i in range(source_pred.size(0)):  # 遍历每个样本
                    if target_indices[i].item() in top_k_indices[i].cpu().numpy():
                        correct_5 += 1
                        
                # 打印调试信息
                if step % 20 == 0:  # 只打印第一个批次的信息
                    print(f"Debug - target_indices shape: {target_indices.shape}, {source_target.shape},top_k_indices shape: {top_k_indices.shape},{source_pred.shape}")
                    print(f"Debug - llm_output shape: {llm_output.shape}")
                    print(f"Debug - Infec_Prob_squeezed shape: {Infec_Prob_squeezed.shape}")
                    #print(f"Debug - Infec_Pop shape: {Infec_Pop[0,:,:]}")
                    # print(f"infec_pop  {infec_pop[0,:,:]}")
                    print(f"Debug - target_indices: {target_indices[:10]}")
                    print(f"Debug - top_k_indices: {top_k_indices[:10]}")
                    print(f"Infec_B:  {Infec_B}")
                    print(f"Infec_N:  {Infec_N}")
                    print(f"Infec_Node_Num: {Infec_Node_Num}")
                    #print(f"batch 0 pop predict:{Infec_Pop[0,:,:]}")
                
                # Hop-1 指标
                preds2 = torch.argmax(source_pred, dim=1).to(device)   # 预测的节点索引 batch_size * 1
                target2 = torch.argmax(source_target, dim=1).to(device)
                correct_Hop_1 += hop_1_metric(contact_graph, preds2, target2)
                


    acc = correct * 100 / test_num
    acc_5 = correct_5 * 100 / test_num
    acc_Hop_1 = correct_Hop_1 * 100 / test_num

    print("========================")
    print(f'Sample_Num:{test_num}')
    print('Top-1 Accuracy: {:.3f}%'.format(acc))
    print('Top-5 Accuracy: {:.3f}%'.format(acc_5))
    print('Hop-1 Accuracy: {:.3f}%'.format(acc_Hop_1))      
    print('Finished! Time used: {:.3f}mins.'.format((time.time() - start_time) / 60))

if __name__ == '__main__':
    main()
