#!/bin/bash

# 获取当前日期和时间
timestamp=$(date +"%Y%m%d_%H%M%S")

# 指定要输出的文件路径
output_file="/home/liubanruo/229codev6/log/check_is_all_zero_$timestamp.out"

# 指定要检查的文件夹路径
folder_path="/home/liubanruo/test_data/data229"

# 遍历文件夹及子文件夹中的所有CSV文件
find "$folder_path" -type f -name "*.csv" ! -name "*diff.csv" | while read -r file_path; do
    # 使用 awk 检查每一列是否全为0
    awk -F',' '
    NR==1 {
        num_columns = NF
    }
    NR>1 {
        all_zero = 1
        for(i=1;i<=NF;i++) {
            if($i!=0) {
                all_zero = 0
                break
            }
        }
        if(all_zero) {
            print FILENAME
            exit
        }
    }' "$file_path" >> "$output_file"
done

