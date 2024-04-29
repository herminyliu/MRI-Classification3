def load_data(data_path, dataset_slice, random_seed, M, sigma, theta):
    import os
    import calculate_Ahat
    import pandas as pd
    import random
    import numpy as np

    SC_normalized_lst = []
    FUN_normalized_lst = []
    feature_matrix_lst = []  # 这个作为特征矩阵！
    diff_matrix_lst = []
    label_lst = []
    label_txt_path = '/home/liubanruo/test_data/control_41270.list'
    with open(label_txt_path, 'r') as file:
        label_contents = file.read()
    id_count = 0
    random.seed(random_seed)
    inidivial_lst = os.listdir(data_path)
    random.shuffle(inidivial_lst)  # 每一个epoch的随机数种子均不一样，打散情况不同。但同一epoch中，随机数种子相同，确保每个被试的数据均使用且仅使用一遍
    for sub_folder in inidivial_lst[dataset_slice]:
        file_path = os.path.join(data_path, sub_folder)
        if search_id_in_file(label_contents, str(sub_folder)):
            label_lst.append(int(0))  # 没患病
        else:
            label_lst.append(int(1))  # 患病了
        for file in os.listdir(file_path):
            SC_temp = np.empty(())
            FUN_temp = np.empty(())
            feature_matrix_temp = np.empty(())
            diff_matrix_temp = np.empty(())
            if file.endswith('_SC_normalized.csv'):
                SC_temp = pd.read_csv(os.path.join(file_path, file), header=None).to_numpy()  # 转换为numpy数组加快计算效率
                if SC_temp.shape != (148, 148):
                    print(f"SC文件的维度不为(148,148), 异常文件地址：{file_path}, 维度：{SC_temp.shape}")
                    continue
                SC_normalized_lst.append(SC_temp)
            if file.endswith('_FUN_normalized.csv'):
                FUN_temp = pd.read_csv(os.path.join(file_path, file), header=None).to_numpy()
               
                if FUN_temp.shape != (148,490):
                    print(f"SC文件的维度不为(148,490), 异常文件地址：{file_path}, 维度：{FUN_temp.shape}")
                    continue
                FUN_normalized_lst.append(FUN_temp)
            if file.endswith('_FUN_corr.csv'):
                feature_matrix_temp = pd.read_csv(os.path.join(file_path, file), header=None).to_numpy()

                if feature_matrix_temp.shape != (148, 148):
                    print(f"feature_matrix文件的维度不为(148,148), 异常文件地址：{file_path}, 维度：{feature_matrix_temp.shape}")
                    continue
                feature_matrix_lst.append(feature_matrix_temp)
            if file.endswith('_FUN_diff.csv'):
                diff_matrix_temp = pd.read_csv(os.path.join(file_path, file), header=None).to_numpy()

                if diff_matrix_temp.shape != (11026, 490):
                    print(f"diff_matrix文件的维度不为(11026,490), 异常文件地址：{file_path}, 维度：{diff_matrix_temp.shape}")
                    continue
                diff_matrix_lst.append(diff_matrix_temp)

        id_count = id_count + 1
        print(f"=======已完成{id_count}个被试者数据的读取=======")

    if len(SC_normalized_lst) == len(FUN_normalized_lst) & len(FUN_normalized_lst) == len(feature_matrix_lst) \
            & len(feature_matrix_lst) == len(diff_matrix_lst):
                print(f"*******SC_normalized_lst FUN_normalized_lst feature_matrix_lst diff_matrix_lst获取完毕，长度均为{len(SC_normalized_lst)}*******")
    else:
        raise ValueError(f"*******列表长度不同*******")

    Af_lst = calculate_Ahat.fun_extraction(FUN_normalized_lst, diff_matrix_lst, M.detach().numpy(), sigma)
    As_lst = SC_normalized_lst

    if len(As_lst) == len(Af_lst) & len(feature_matrix_lst) == len(Af_lst):
        print(f"*******As_lst Af_lst feature_matrix_lst正常生成，列表长度均为：{len(Af_lst)}*******")
    else:
        raise ValueError(f"*******As_lst和Af_lst列表长度不同，As_lst长度为：{len(As_lst)}，Af_lst长度为：{len(Af_lst)}，feature_matrix长度为：{len(feature_matrix_lst)}")
    Ahat_lst = []  # 这个作为连接矩阵！
    fusion_count = 0
    for i in range(len(Af_lst)):
        Ahat, is_nan = calculate_Ahat.fusion(theta.item(), Af_lst[i], As_lst[i])
        Ahat_lst.append(Ahat)
        if is_nan == 0:
            print(f"=======已完成{i}个Ahat矩阵的融合生成=======")
        else:
            print(f"=======已完成{i}个Ahat矩阵的融合生成，该批Ahat_lst中的第{i+1}Ahat出现{is_nan}个NaN值=======")
    print(f"*******Af_lst获取完毕，长度为{len(Af_lst)}*******")

    dataset = []
    for i in range(len(Ahat_lst)):
        dataset.append(bulid_single_graph(Ahat_lst[i], feature_matrix_lst[i], label_lst[i]))  # 这里还没有label
        print(f"=======已完成{i}个被试者脑图的生成=======")
    print(f"*******Ahat_lst获取完毕，长度为{len(Ahat_lst)}*******")

    # 返回列表，列表中的每个元素均是一个图
    return dataset


def search_id_in_file(file_contents, target_id):
    for line in file_contents.split('\n'):
        # 去除行尾的换行符并比较ID
        if line.strip() == target_id:
            return True
    return False


def bulid_single_graph(adjacency_matrix, node_features, is_ill):
    import torch
    from torch_geometric.data import Data
    # 假设连接关系矩阵存储在一个名为 'adjacency_matrix.csv' 的CSV文件中
    # 获取连接矩阵的行数，即节点数量
    num_nodes = adjacency_matrix.shape[0]

    # 创建边的列表，创建边的权重列表
    edges = []
    edge_weights = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] != 0:
                edges.append((i, j))
                edges.append((j, i))  # 很重要！如果只有ij没有ji将是有向图！
                edge_weights.append(adjacency_matrix[i, j])
                edge_weights.append(adjacency_matrix[i, j])  # 必须是两行！因为这个是无向图, [i,j]和[j,i]一样

    # 将python列表转换成张量
    x = torch.tensor(node_features, dtype=torch.float)  # 节点特征
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 边的索引
    edge_weights = torch.tensor(edge_weights, dtype=torch.long).t().contiguous()  # 边的权重

    # 打印成功转换信息，并且返回pyG的graph格式
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=is_ill)
