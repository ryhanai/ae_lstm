# -*- coding: utf-8 -*-

import os, sys, glob, re, time, copy
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils import *
import model, trainer


os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


dataset='reaching'
train_groups=range(1,300)
val_groups=range(300,350)
input_image_size=(80,160)
time_window_size=20
latent_dim=32
dof=7

model = model.model_ae_lstm(input_image_size+(3,), time_window_size, latent_dim, dof)

def train():
    train_ds = Dataset(dataset)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, train_ds, val_ds, time_window_size=time_window_size)
    tr.train()
    return tr

def prepare_for_test(cp='ae_cp.reaching.ae_lstm.20220112181145'):
    val_ds = Dataset(dataset)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

# tr.predict_images()
# tr.predict_joint_angles()
# tr.predict_sequence_closed(0)
# tr.predict_for_group(0)
