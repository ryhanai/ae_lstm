#!/bin/bash

# ipython test_torch.py -i -- \
#     --dataset_path ~/Dataset/forcemap\
#     --task_name tabletop_airec241008\
#     --data_split test\
#     --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20250125_2313_34\
#     --weight_file 00080.pth


ipython test_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec241008\
    --data_split test\
    --weights ~/Program/moonshot/ae_lstm/scripts/force_estimation/log/20241017_0052_35/\
    --weight_file 08000.pth
