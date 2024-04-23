import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"C:\Users\28760\Desktop\脑图大创\24年新\229服务器数据\data229\1000186\1000186_FUN_normalized_FUN_diff.csv", header=None, index_col=None)

# 减少小数位数
df = df.round(decimals=8)  # 将所有数字保留3位小数，你可以根据需要调整这个数字

# 保存文件
df.to_csv(r'C:\Users\28760\Desktop\脑图大创\24年新\229服务器数据\data229\1000186\compressed_file.csv', index=False, header=False, float_format='%.6e')
