#!/bin/bash

for seed in {2..2}; do
    echo "seed = $seed"
    python train_torch.py \
        --dataset_path ~/Dataset/forcemap\
        --task_name tabletop250902\
        --model fmdev.force_estimation_v4.ForceEstimationResNetTabletop\
        --epoch 200\
        --batch_size 16\
        --seed $seed\
        --lr 1e-3\
        --method 'geometry-aware'\
        --sigma_f 0.03\
        --sigma_g 0.01
done
