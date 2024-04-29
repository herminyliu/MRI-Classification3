import os

def list_subdirectories(folder_path, output_file):
    # 检查给定路径是否是一个文件夹
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a directory.")
        return
    
    # 使用 os.listdir 获取文件夹下所有的文件和文件夹名字
    items = os.listdir(folder_path)
    
    # 存储子文件夹名字的列表
    subdirectories = []
    
    # 遍历文件夹下的所有项目
    for item in items:
        item_path = os.path.join(folder_path, item)
        # 检查项目是否是一个文件夹
        if os.path.isdir(item_path):
            subdirectories.append(item)
            print(item)
    
    # 将子文件夹名字按顺序写入输出文件
    with open(output_file, 'w') as f:
        for subdir in subdirectories:
            f.write(subdir + '\n')

# 指定要列出子文件夹的文件夹路径
folder_path = "/home/liubanruo/test_data/data229"
# 指定输出文件的路径
output_file = "/home/liubanruo/test_data/listID_in_order.list"

# 列出文件夹下所有子文件夹的名字，并将它们按顺序记录在一个 .list 文件中
list_subdirectories(folder_path, output_file)

