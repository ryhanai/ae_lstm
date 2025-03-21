#!/bin/bash

for seed in {1..5}; do
    echo "seed = $seed"
    python train_torch.py \
        --dataset_path ~/Dataset/forcemap\
        --task_name tabletop240304\
        --model fmdev.force_estimation_v4.ForceEstimationResNetTabletop\
        --epoch 200\
        --batch_size 16\
        --seed $seed\
        --lr 1e-3\
        --method 'isotropic'\
        --sigma_f 0.015\
        --sigma_g 0.01
done
