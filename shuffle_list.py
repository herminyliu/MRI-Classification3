import random

# 读取原始文件
with open('/home/liubanruo/test_data/listID_in_order.list', 'r') as file:
    lines = file.readlines()

# 搅乱列表
random.seed(42)
random.shuffle(lines)

# 写入新文件
with open('/home/liubanruo/test_data/original_list_shuffle.list', 'w') as file:
    file.writelines(lines)

