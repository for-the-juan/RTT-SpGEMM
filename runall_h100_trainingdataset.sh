#!/bin/bash

DATASET_DIR="/data/home/scwb383/run/training_dataset"
LOG_DIR="/data/home/scwb383/run/TileSpGEMM_log/training_dataset"

mkdir -p "$LOG_DIR"

# 使用find遍历所有.mtx文件（包括子文件夹）
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    # 提取矩阵文件名（不含扩展名）
    base_name_=$(basename $mtx_file .mtx)
    ./src/test -d 0 -aat 0 $mtx_file > /data/home/scwb383/run/TileSpGEMM_log/training_dataset/${base_name_}.log
done

echo "所有测试完成!"