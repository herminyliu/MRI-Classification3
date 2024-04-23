import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 定义数据预处理和数据加载器
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化theta为一个矩阵，形状与MNIST图像相同
theta = torch.randn(28, 28, requires_grad=True)
epochs = 10

best_loss = float('inf')
best_theta = None

# 训练循环
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # 调整图像灰度
        data_adjusted = data * theta

        output = model(data_adjusted)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 在每个epoch结束后评估模型性能
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_adjusted = data * theta
            output = model(data_adjusted)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    # 根据性能调整theta
    if test_loss < best_loss:
        best_loss = test_loss
        best_theta = theta.clone().detach()
    else:
        # 减小theta以降低灰度，但保留一些像素的权重
        theta.data -= 0.01 * torch.sign(theta.grad)

# 最佳灰度参数
best_theta = best_theta.detach().numpy()
print("Best Theta:", best_theta)
