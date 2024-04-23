import numpy as np


# 定义主函数
def fun_extraction(FUN_normalized_lst, diff_matrix_lst, M, sigma):
    af_matrices_lst = []
    Af_count = 0
    for i in range(len(FUN_normalized_lst)):
        Af = calculate_Af(FUN_normalized_lst[i], diff_matrix_lst[i], M, sigma)
        af_matrices_lst.append(Af)
        Af_count = Af_count + 1
        print(f"=======已完成{Af_count}个Af矩阵的生成=======")
    return af_matrices_lst


def calculate_Af(FUN_matrix_normalized, diff_matrix, M, sigma):
    # 获取行数
    row_num = FUN_matrix_normalized.shape[0]

    # 初始化 Af 矩阵
    Af = np.zeros((row_num, row_num))

    # 计算 Af 矩阵的每个元素
    count_raw = 0
    for i in range(row_num):
        for j in range(i, row_num):
            M_diff_squared = np.dot(M, diff_matrix[count_raw, :])  # 两个向量相乘结果是一个标量。147 * i + j反推ij脑区的diff行数
            Af[i, j] = np.exp(M_diff_squared / (-2 * sigma ** 2))  # M_diff_squared向量元素平均
            Af[j, i] = Af[i, j]  # Af 矩阵是对称的，所以对称位置也要赋值
            count_raw = count_raw + 1
    return Af


def fusion(theta, Af, As):
    # 计算 beta1 和 beta2
    beta1 = np.log(theta / (1 - theta))
    beta2 = -beta1

    # 计算 theta1 和 theta2
    theta1 = np.exp(-beta1) / (np.exp(-beta1) + np.exp(beta2))
    theta2 = np.exp(-beta2) / (np.exp(-beta1) + np.exp(beta2))

    # 计算 A
    I = np.eye(len(Af))
    Ahat = theta1 * Af + theta2 * As + I
    return Ahat
