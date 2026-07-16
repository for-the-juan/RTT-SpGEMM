#!/bin/bash

# AAT="/home/stu1/Dataset/simple"
AA="/data/home/scwb383/run/TileSpGEMMDataset"

mkdir -p /data/home/scwb383/run/TileSpGEMM_log
mkdir -p /data/home/scwb383/run/TileSpGEMM_log/TileSpGEMMDataset
mkdir -p /data/home/scwb383/run/TileSpGEMM_log/HYTEDataset
for mtx_file in "$AA"/*.mtx; do
    base_name_=$(basename $mtx_file .mtx)
    ./src/test -d 0 -aat 0 $mtx_file > /data/home/scwb383/run/TileSpGEMM_log/TileSpGEMMDataset/${base_name_}.log
    echo "${base_name_} Finished!"
done