import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def z_score_normalize(df):
    # 使用z-score标准化
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df.T).T  # 按行进行标准化
    return pd.DataFrame(standardized_data, columns=df.columns)


def log_transform(df):
    # 将表中的每个元素进行对数转化
    return np.log10(df + 1)


def process_csv_file(csv_file_path, subdir_path):
    df = pd.read_csv(csv_file_path, header=0, index_col=0)
    df = log_transform(df)
    sliced_df = df.iloc[0:148, 0:148]
    # 标准化数据
    standardized_df = z_score_normalize(sliced_df.astype(float))
    # 构建新的文件名
    new_file_name = os.path.splitext(os.path.basename(csv_file_path))[0] + '_normalized.csv'
    new_file_path = os.path.join(subdir_path, new_file_name)
    # 将标准化后的数据保存为新的CSV文件，注意我们不要列名和索引
    standardized_df.to_csv(new_file_path, index=False, header=False)
    print(f"{new_file_name}已成功保存")


def process_folder(folder_path):
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            # 处理_SC.csv文件和_FUN.csv文件
            csv_files = [file for file in os.listdir(subdir_path) if file.endswith('_SC.csv')]
            for csv_file in csv_files:
                csv_file_path = os.path.join(subdir_path, csv_file)
                # 判断当前文件是否为_SC.csv文件
                process_csv_file(csv_file_path, subdir_path)


if __name__ == "__main__":
    # !!!注意，这是按行进行的标准化
    process_folder(r"/home/liubanruo/test_data/data229")
