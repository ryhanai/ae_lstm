#!/bin/bash

ipython3 test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop240304\
    --data_split test\
    --weight_files log/20250321_0141_06/00199.pth


# GAFS 0.03,0.01: log/20250321_0935_35/00199.pth
# GAFS 0.06,0.01: log/20250321_0141_06/00199.pth
# IFS 0.015: log/20250321_1423_23/00199.pth
# IFS 0.005: log/20250321_1243_40/00199.pth