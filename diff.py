import numpy as np


def diff(FUN_matrix):
    # 获取行数
    num_brain_area = FUN_matrix.shape[0]
    num_timepoint = FUN_matrix.shape[1]
    diff_2d_matrix = np.zeros((int(num_brain_area * num_brain_area / 2 + num_brain_area / 2), int(num_timepoint + 2)))
    row_count = 0
    for i in range(num_brain_area):
        for j in range(i, num_brain_area):
            diff = FUN_matrix.iloc[i, :] - FUN_matrix.iloc[j, :]
            diff_squared = diff ** 2
            diff_2d_matrix[row_count, :] = diff_squared
            row_count = row_count + 1

    return diff_2d_matrix


if __name__ == "__main__":
    import os
    import pandas as pd

    data_path = r'C:\Users\28760\Desktop\脑图大创\24年新\229服务器数据\data229'

    for sub_folder in os.listdir(data_path):
        file_path = os.path.join(data_path, sub_folder)
        for file in os.listdir(file_path):
            if file.endswith('_FUN_normalized.csv'):
                FUN_temp = pd.read_csv(os.path.join(file_path, file), header=None, index_col=None)
                diff_matrix = diff(FUN_temp)
                new_name = file.split('.')[0] + '_FUN_diff.csv'
                # 保存二维数组为 .csv 文件
                np.savetxt(os.path.join(file_path, new_name), diff_matrix, delimiter=',')
                print(f"{new_name}已保存")

