import numpy as np
import pandas as pd

# 定义 calculate_Af 函数
def calculate_Af(FUN_matrix_normalized, M, sigma):
    diff = FUN_matrix_normalized.values[:, :, np.newaxis] - FUN_matrix_normalized.values[:, np.newaxis, :]
    diff_squared = diff ** 2
    M_diff_squared = np.dot(M[:, np.newaxis, np.newaxis], diff_squared)  # 正确执行向量化乘法
    Af = np.exp(M_diff_squared / (-2 * sigma ** 2))
    np.fill_diagonal(Af, 0)
    return Af

# 生成示例数据
np.random.seed(0)
FUN_matrix_normalized = pd.DataFrame(np.random.rand(148, 148))  # 示例 FUN_matrix_normalized
M = np.random.rand(490)  # 示例 M 向量
sigma = 1.0  # 示例 sigma

# 调用函数计算 Af
Af = calculate_Af(FUN_matrix_normalized, M, sigma)

# 打印结果
print("Af 矩阵:")
print(Af)
