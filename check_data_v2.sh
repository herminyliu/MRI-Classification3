#!/bin/bash

# 检查给定路径是否存在
if [ ! -d "$1" ]; then
  echo "Error: Directory not found!"
  exit 1
fi

# 获取当前的月份、日期、小时和分钟
current_date=$(date +"%m%d%H%M")

# 生成输出文件名
output_file="log/check_data$current_date.out"

# 遍历给定路径下的所有子文件夹
for dir in "$1"/*; do
  # 获取文件夹名，即ID
  id=$(basename "$dir")
  
  # 检查是否是7位数ID
  if [[ ! $id =~ ^[0-9]{7}$ ]]; then
    continue
  fi
  echo $id  
  # 检查子文件夹是否缺少这六个文件
  missing_files=0
  # for filename in "${id}"_FUN.csv "${id}"_SC.csv "${id}"_FUN_normalized_FUN_corr.csv "${id}"_FUN_normalized_FUN_diff.csv "${id}"_FUN_normalized.csv "${id}"_SC_normalized.csv . ..; do
  for filename in "${id}"{"_FUN.csv", "_SC.csv", "_FUN_normalized_FUN_corr.csv", "_FUN_normalized_FUN_diff.csv", "_FUN_normalized.csv", "_SC_normalized.csv"}; do  
    if [ ! -f "$dir/$filename" ]; then
      missing_files=1
      break
    fi
  done
  
  if [ $missing_files -eq 1 ]; then
    echo "Missing files in $dir" >> "$output_file"
  fi

echo "Done! Check "$output_file" for details."

