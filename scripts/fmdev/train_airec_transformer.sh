#!/bin/bash

ipython train_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec250125\
    --model ForceEstimationV5\
    --epoch 301\
    --batch_size 8\
    --lr '1e-4'\
    --seed 0\
    --method 'geometry-aware'