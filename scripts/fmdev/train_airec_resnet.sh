#!/bin/bash

python train_torch.py \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec241008\
    --model ForceEstimationResNetTabletop\
    --epoch 100\
    --batch_size 16\
    --seed 1\
    --lr 1e-3\
    --method 'geometry-aware'\
    --sigma_f 0.03\
    --sigma_g 0.01