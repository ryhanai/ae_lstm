#!/bin/bash

ipython train_torch.py -i -- \
    --dataset_path ~/Dataset/forcemap\
    --task_name tabletop_airec250125_4000\
    --model ForceEstimationResNetTabletop\
    --epoch 2000\
    --batch_size 8\
    --seed 1\    
    --method 'geometry-aware'