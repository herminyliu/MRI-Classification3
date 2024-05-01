import pandas as pd
import torch
import numpy as np

# 定义预测值张量
pred = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# 定义真实值张量
true_value = torch.tensor([1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])

# 输出预测值和真实值
print("预测值(pred):", pred)
print("真实值(true value):", true_value)

# correct = int(np.sum(pred == true_value))
# 我真的绷不住了！！这一行真的害人不浅！！
# 在你的代码中，你使用了 torch.tensor 来创建张量，但是在计算正确预测数量时，你使用了 NumPy 的 np.sum() 函数，这可能导致了类型不匹配的错误。虽然这两个张量的长度相同，但是它们的类型不同，一个是 PyTorch 张量，一个是 NumPy 数组。
correct = torch.sum(pred == true_value).item()
print(len(pred))
print(len(true_value))
print(correct)
