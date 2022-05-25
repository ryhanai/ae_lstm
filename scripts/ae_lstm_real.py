# -*- coding: utf-8 -*-

import os

from core.utils import *
#from core.model import *
from model import *
import trainer


os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


dataset='reaching-real'
train_groups=range(0,136)
val_groups=range(136,156)
joint_range_data=range(0,156)
input_image_size=(80,160)
time_window_size=20
latent_dim=64
dof=7

model_ae = model_autoencoder(input_image_size+(3,), latent_dim)
model_ae_lstm = model_ae_lstm_aug(input_image_size+(3,), time_window_size, latent_dim, dof, joint_noise=0.03)

def train_ae():
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_ae, train_ds, val_ds)
    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)
    return tr

def train_ae_lstm(ae_cp='ae_cp.reaching-real.autoencoder.20220517112615'):
    """
    train AE+LSTM jointly using pre-trained weight of AutoEncoder
    """
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)

    # load AE weights and copy them to the encoder and decoder of AE-LSTM model
    d = os.path.join(os.path.dirname(os.getcwd()), 'runs')
    model_ae.load_weights(os.path.join(d, ae_cp, 'cp.ckpt'))
    model_ae_lstm.get_layer('encoder').set_weights(model_ae.get_layer('encoder').get_weights())
    model_ae_lstm.get_layer('decoder').set_weights(model_ae.get_layer('decoder').get_weights())

    tr = trainer.TimeSequenceTrainer(model_ae_lstm, train_ds, val_ds, time_window_size=time_window_size)
    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)

def prepare_for_test_ae(cp='ae_cp.reaching-real.autoencoder.20220517112615'):
    # ae_cp.reaching-real.autoencoder.20220516143855 # no augmentation
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_ae, None, val_ds, checkpoint_file=cp)
    return tr

def prepare_for_test(cp='ae_cp.reaching-real.ae_lstm_aug.20220524225920'):
    # ae_cp.reaching-real.ae_lstm_aug.20220524193755 # AE pretraining, 5step, no random_translation
    # ae_cp.reaching-real.ae_lstm_aug.20220524180842 # AE pretraining, 3step, no random_translation
    # ae_cp.reaching-real.ae_lstm_aug.20220517230019 # AE pretraining, random_translation
    # ae_cp.reaching-real.ae_lstm_aug.20220517145720 # AE pretraining, no augmentation
    # 20220426162642 # brightness-contrast-hue(0.2)
    # 20220426132636 # contrast augmentation
    # 20220426103749 # frame-wise random translation
    # 20220420174439
    # 20220419144918
    # 20220414215231
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.TimeSequenceTrainer(model_ae_lstm, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

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

def prepare_for_test2(cp='ae_cp.reaching-real.ae_lstm.20220419144918'):
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, train_ds, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

def test_augment(batch, brightness_max_delta=0.2, contrast_lower=0.8, contrast_upper=1.2, hue_max_delta=0.05):
    imgs = batch[0][0][:,-1]
    aug_imgs = augment(imgs, brightness_max_delta=brightness_max_delta, contrast_lower=contrast_lower, contrast_upper=contrast_upper, hue_max_delta=hue_max_delta)
    visualize_ds(aug_imgs)
    plt.show()

from PIL import ImageSequence

def visualize_frames_from_anim_gif(gif_file):
    im = Image.open(gif_file)
    frames = list(frame.copy() for frame in ImageSequence.Iterator(im))
    visualize_ds(frames)
    plt.show()


#train_ae_lstm()

# Training
# In[1]: train()

# Test
# In[1]: prepare_for_test()
# In[2]: tr.predict_images()
# In[3]: tr.predict_joint_angles()
# In[4]: tr.predict_sequence_closed(0)
# In[5]: tr.predict_for_group(0)
