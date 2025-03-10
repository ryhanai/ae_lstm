#!/bin/bash

ipython test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec250125\
    --data_split test\
    --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20250309_1520_50\
    --weight_file 00300.pth
