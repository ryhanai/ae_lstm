#!/bin/bash

ipython test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop240125\
    --data_split test\
    --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20240304_1834_24\
    --weight_file ForceEstimationResNetTabletop.pth
