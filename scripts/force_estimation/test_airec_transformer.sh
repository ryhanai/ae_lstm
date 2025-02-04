#!/bin/bash

ipython test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec250125\
    --data_split test\
    --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20250204_1849_08\
    --weight_file 00100.pth
