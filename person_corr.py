def person_corr(FUN_matrix):
    import numpy as np
    num_rows, num_cols = FUN_matrix.shape
    result_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(num_rows):
            if i == j:
                result_matrix[i, j] = 1.0  # Pearson correlation of a variable with itself is 1
            else:
                correlation = np.corrcoef(FUN_matrix[i], FUN_matrix[j])[0, 1]
                result_matrix[i, j] = correlation

    return result_matrix


if __name__ == "__main__":
    import os
    import pandas as pd
    print(123)
    data_path = r'C:\Users\28760\Desktop\脑图大创\24年新\229服务器数据\data229'

    for sub_folder in os.listdir(data_path)[450:]:
        file_path = os.path.join(data_path, sub_folder)
        for file in os.listdir(file_path):
            if file.endswith('_FUN_normalized.csv'):
                FUN_temp = pd.read_csv(os.path.join(file_path, file), header=None, index_col=None)
                corr_matrix = person_corr(FUN_temp)
                corr_matrix = pd.DataFrame(corr_matrix)
                new_name = file.split('.')[0] + '_FUN_corr.csv'
                corr_matrix.to_csv(os.path.join(file_path, new_name), header=False, index=False)
                print(f"{new_name}已保存")