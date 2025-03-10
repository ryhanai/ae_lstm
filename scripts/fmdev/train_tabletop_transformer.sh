#!/bin/bash

ipython train_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop240125\
    --model ForceEstimationV5\
    --epoch 1000\
    --batch_size 8\
    --method 'geometry-aware'
