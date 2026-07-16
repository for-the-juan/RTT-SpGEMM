#!/bin/bash

DATASET_DIR="/data/home/scwb383/run/training_dataset"
LOG_DIR="/data/home/scwb383/run/TileSpGEMM_log/training_dataset_h200"

mkdir -p "$LOG_DIR"

TIMEOUT_SECONDS=45

find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "运行: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./src/test -d 0 -aat 0 "$mtx_file" > "$LOG_DIR/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "警告: ${base_name}_aat0_m${m}_n${n} 运行超时（超过$(($TIMEOUT_SECONDS/60))分钟），已跳过"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "警告: ${base_name}_aat0_m${m}_n${n} 运行失败，退出码: $exit_code"
        continue
    fi
done

echo "所有测试完成!"