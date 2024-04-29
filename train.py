import numpy as np
import random
import torch
from data_loader import load_data
import model


def train(my_model_train, train_loader):
    my_model_train.train()
    total_loss = 0
    loss_lst = []
    for _, train_data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        out = my_model_train(train_data.x, train_data.edge_index, train_data.batch)  # Perform a single forward pass.
        loss = criterion(out, train_data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss = total_loss + loss
        loss_lst.append(loss)
    return total_loss, loss_lst


def valid(my_model_valid, valid_loader):
    my_model_valid.eval()
    total_loss = 0
    correct = 0
    for _, valid_data in enumerate(valid_loader):  # Iterate in batches over the training/test dataset.
        out = my_model_valid(valid_data.x, valid_data.edge_index, valid_data.batch)
        total_loss += criterion(out, valid_data.y).item()
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int(np.sum(pred == valid_data.y))  # Check against ground-truth labels.

    return correct / len(valid_loader.dataset), total_loss  # Derive ratio of correct predictions.


def shuffle_dataset(input_list, seed):
    random.seed(seed)  # 设置随机数种子
    shuffled_dataset = input_list[:]  # 复制输入列表，以避免直接修改原始列表
    random.shuffle(shuffled_dataset)  # 打乱列表
    return shuffled_dataset


def split_dataset(dataset):
    train_dataset = dataset[:round(len(dataset) * train_ratio)]
    valid_dataset = dataset[round(len(dataset) * train_ratio):]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(valid_dataset)}')

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset)+1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset)+1, shuffle=False)

    return train_loader, valid_loader


def one_batch_train(dataset_slice, random_seed):
    from torch_geometric.loader import DataLoader
    dataset_lst = load_data(data_path, dataset_slice, random_seed, M, sigma, theta)
    train_loader = DataLoader(dataset_lst, batch_size=len(dataset_lst), shuffle=False)  # dataloader.py里已经打散过了
    return train(my_model_train=my_model, train_loader=train_loader)


def one_batch_valid(random_seed):
    from torch_geometric.loader import DataLoader
    dataset_lst = load_data(data_path, slice(950, 999), random_seed, M, sigma, theta)  # 稳定选择最后49个作为每个epoch后的验证集,有一个被试者缺少FUN数据
    valid_loader = DataLoader(dataset_lst, batch_size=len(dataset_lst), shuffle=False)  # dataloader.py里已经打散过了
    return valid(my_model_valid=my_model, valid_loader=valid_loader)


def one_epoch(epoch, best_loss, random_seed):
    print(f"***************第{epoch+1}轮训练开始***************")
    train_loss_one_epoch = 0
    for i in range(0, int(dataset_total_length/batch_size) - 1):  # -1是为了留一个训练集
        print(f"***************第{epoch+1}轮{i+1}批训练开始***************")
        train_loss_one_batch, loss_lst = one_batch_train(slice(i*50,  (i+1)*50), random_seed)
        train_loss_one_epoch = train_loss_one_epoch + train_loss_one_batch
        print(f'Epoch: {epoch:01d}, Batch: {i+1:01d}, Train Loss:{train_loss_one_batch:.3f}')
        print(f'Epoch: {epoch:01d}, Batch: {i+1:01d}, Loss List:{[tensor.item() for tensor in loss_lst]}')

    valid_loss_one_epoch, acc_one_epoch = one_batch_valid(random_seed)
    train_loss_lst.append(train_loss_one_epoch)
    valid_loss_lst.append(valid_loss_one_epoch)
    valid_acc_lst.append(acc_one_epoch)

    print(f'Epoch: {epoch:03d},Train Loss:{valid_loss_one_epoch:.3f},'
          f'Test Loss:{valid_loss_one_epoch:.3f},Test Acc:{acc_one_epoch:.3f}')
    # 这里需要迭代参数M和theta
    if valid_loss_one_epoch < best_loss:
        best_loss = valid_loss_one_epoch
        print(f"======best_loss更新，新的值为:{best_loss}========")
    else:
        # 减小theta以降低灰度，但保留一些像素的权重
        M.data -= learning_rate * torch.sign(M.grad)
        theta.data -= learning_rate * torch.sign(theta.grad)

    return best_loss


def save_fig(train_acc_lst, test_acc_lst):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 设置seaborn样式
    sns.set(style="whitegrid")
    # 创建索引列表（横轴）
    indices = list(range(len(train_acc_lst)))
    # 创建 Seaborn 绘图
    sns.lineplot(x=indices, y=train_acc_lst, label="Train Accuracy")
    sns.lineplot(x=indices, y=test_acc_lst, label="Test Accuracy")

    # 添加标题和标签
    plt.title("Accuracy of Graph Classification")
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")

    # 显示图例
    plt.legend()
    file_path = './Result_Figure/Accuracy_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.png'
    print(f"*******图片已成功保存在：{file_path}")
    plt.savefig(file_path, dpi=600)


if __name__ == '__main__':
    data_path = r'/home/liubanruo/test_data/data229'
    theta = torch.tensor(0.5, requires_grad=True)  # 这个参数需要迭代
    best_loss = 1e10
    M = torch.ones(490, requires_grad=True)  # 490个时间戳，这个参数需要迭代，注意为了简化这里变成了向量
    sigma = 1  # 超参数，不需要训练中迭代
    seed_value = 42
    train_ratio = 0.7
    test_ratio = 1 - train_ratio
    learning_rate = 0.0005  # 超参数
    weight_decay = 0.01  # 超参数
    batch_size = 50
    epoches_to_run = 50
    dataset_total_length = 999  # 原本为1000，但有一个被试缺少FUN数据
    train_acc_lst = []
    train_loss_lst = []
    valid_acc_lst = []
    valid_loss_lst = []

    # 模型
    my_model = model.GCN(dataset_num_node_features=148, hidden_channels=148)
    # 优化器
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 开始使用模型
    for epoch in range(0, epoches_to_run):
        best_loss = one_epoch(epoch=epoch, best_loss=best_loss, random_seed=seed_value)
        seed_value = seed_value + 1

    import datetime
    the_datetime_of_run = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    save_fig(train_acc_lst, valid_acc_lst)

    # 保存模型（整个模型）
    my_model.eval()
    torch.save(model, './Result_Model_Para/pyg_model_' + the_datetime_of_run + '.pth')

    # 定义一个文件名来保存输出信息
    output_file = "./Result_Model_Para/model_summary_" + the_datetime_of_run + ".txt"
    # 将输出信息写入到txt文件
    with open(output_file, "w") as f:
        f.write(str(model) + '\n')
        f.write('训练优化器为：\n' + str(optimizer) + '\n')
        f.write('训练损失函数为：\n' + str(criterion) + '\n')
        f.write('测试集训练集打乱种子为：' + str(seed_value) + '\n')
        f.write("训练轮数为：" + str(len(train_acc_lst)+1) + '\n')
        f.write("训练损失为：\n")
        f.write(str(train_loss_lst) + '\n')
        f.write("训练准确率为：\n")
        f.write(str(train_acc_lst) + '\n')
        f.write("测试损失为：\n")
        f.write(str(valid_loss_lst) + '\n')
        f.write("测试准确率为：\n")
        f.write(str(valid_acc_lst) + '\n')
        f.write("程序训练时间为：" + the_datetime_of_run + '\n')
        f.write("程序作者为：" + "刘般若")
    f.close()

    print(f"模型信息已保存到 {output_file},程序成功执行完毕")
