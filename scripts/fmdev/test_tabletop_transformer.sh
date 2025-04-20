#!/bin/bash

ipython3 test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop240304\
    --data_split test\
    --weight_files "log/20250322_1023_08/00199.pth log/20250419_1648_52/00199.pth log/20250419_2043_18/00199.pth"
# ResNet + MSE: log/20250322_1023_08/00199.pth
# Transformer + MSE: log/20250419_1648_52/00199.pth
# TRansformer + PCL: log/20250419_2043_18/00199.pth