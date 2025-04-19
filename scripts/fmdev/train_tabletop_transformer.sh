#!/bin/bash

for seed in {1..1}; do
    echo "seed = $seed"
    python train_torch.py \
        --dataset_path ~/Dataset/forcemap\
        --task_name tabletop240304\
        --model fmdev.force_estimation_v5.ForceEstimationV5\
        --epoch 200\
        --batch_size 8\
        --seed $seed\
        --lr 1e-3\
        --loss pcl\
        --method 'geometry-aware'\
        --sigma_f 0.03\
        --sigma_g 0.01        
done
