#!/bin/bash

ipython test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec241008\
    --data_split test\
    --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20250124_2054_23\
    --weight_file best.pth
