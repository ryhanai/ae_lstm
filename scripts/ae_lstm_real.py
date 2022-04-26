# -*- coding: utf-8 -*-

import os

from core.utils import *
#from core.model import *
from model import *
from core import trainer


os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


dataset='reaching-real'
train_groups=range(0,136)
val_groups=range(136,156)
joint_range_data=range(0,156)
input_image_size=(80,160)
time_window_size=20
latent_dim=32
dof=7

#model = model_ae_lstm(input_image_size+(3,), time_window_size, latent_dim, dof, joint_noise=0.03)
model = model_ae_lstm_aug(input_image_size+(3,), time_window_size, latent_dim, dof, joint_noise=0.03)

def train():
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, train_ds, val_ds, time_window_size=time_window_size)
    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)
    return tr

def prepare_for_test(cp='ae_cp.reaching-real.ae_lstm.20220426162642'):
    # 20220426132636 # contrast augmentation
    # 20220426103749 # frame-wise random translation
    # 20220420174439
    # 20220419144918
    # 20220414215231
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

def prepare_for_test2(cp='ae_cp.reaching-real.ae_lstm.20220419144918'):
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, train_ds, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

# Training
# In[1]: train()

# Test
# In[1]: prepare_for_test()
# In[2]: tr.predict_images()
# In[3]: tr.predict_joint_angles()
# In[4]: tr.predict_sequence_closed(0)
# In[5]: tr.predict_for_group(0)
