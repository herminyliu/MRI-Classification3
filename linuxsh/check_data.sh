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
  
  # 检查子文件夹是否缺少这六个文件
  missing_files=0
  # for filename in "${id}"_{FUN,SC}{,_normalized,_normalized_FUN_{corr,diff}}.csv; do
  for filename in "${id}"{"_FUN.csv", "_SC.csv", "_FUN_normalized_FUN_corr.csv", "_FUN_normalized_FUN_diff.csv", "_FUN_normalized.csv", "_SC_normalized.csv"}; do  
    if [ ! -f "$dir/$filename" ]; then
      missing_files=1
      break
    fi
  done
  
  if [ $missing_files -eq 1 ]; then
    echo "Missing files in $dir" >> "$output_file"
  fi
  
  # 遍历子文件夹中的所有文件
  for file in "$dir"/*; do
    # 检查文件是否符合命名规则
    if [[ $file =~ /([0-9]{7})_(FUN|SC)(_normalized)?(_FUN_(corr|diff))?\.csv$ ]]; then
      # 检查文件是否含有NaN值
      if grep -q NaN "$file"; then
        echo "Replacing NaN with 0 in $file" >> "$output_file"
        # 将NaN值替换为0
        sed -i 's/NaN/0/g' "$file"
      fi
    fi
  done
done

echo "Done! Check "$output_file" for details."

