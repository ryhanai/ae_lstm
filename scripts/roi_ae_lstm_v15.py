# -*- coding: utf-8 -*-

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from core.utils import *
from core.model import *
from core import trainer

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


dataset = 'peg-in-hole-no-shadow'
train_groups=range(1,300)
val_groups=range(300,350)
input_image_size=(80,160)
time_window_size=20
latent_dim=32
dof=7

roi_size = (40, 80)


def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, input_image_size)

def roi_rect(args):
    c, s = args
    lt = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c
    rb = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c + tf.tile(tf.expand_dims(s, 1), (1,2))
    roi = tf.concat([lt, rb], axis=1)
    roi3 = tf.expand_dims(roi, 0)
    return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

def roi_rect1(args):
    c, s = args
    lt = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c
    rb = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c + tf.tile(tf.expand_dims(s, 1), (1,2))
    return tf.concat([lt, rb], axis=1)
    
def embed(args):
    whole_images, roi_images, rois = args
    roi = rois[0][0]
    y1 = roi[0]
    x1 = roi[1]
    y2 = roi[2]
    x2 = roi[3]
    y = tf.cast(input_image_size[0] * y1, tf.int32)
    x = tf.cast(input_image_size[1] * x1, tf.int32)
    h = tf.cast(input_image_size[0] * (y2 - y1), tf.int32)
    w = tf.cast(input_image_size[1] * (x2 - x1), tf.int32)
    resized_roi_images = tf.image.resize(roi_images, (h,w))
    padded_roi_images = tf.image.pad_to_bounding_box(resized_roi_images, y, x, input_image_size[0], input_image_size[1])

    d = 1.0 # dummy
    mask = (resized_roi_images + d) / (resized_roi_images + d)
    fg_mask = tf.image.pad_to_bounding_box(mask, y, x, input_image_size[0], input_image_size[1])
    bg_mask = 1.0 - fg_mask
    merged_img = fg_mask * padded_roi_images + bg_mask * whole_images
    return merged_img

class AutoencoderWithCrop(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(AutoencoderWithCrop, self).__init__(*args, **kargs)
        tracker_names = ['image_loss', 'joint_loss', 'scale', 'loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            image_loss, joint_loss, scale, loss = self.compute_loss(y, y_pred, x)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['image_loss'].update_state(image_loss)
        self.train_trackers['joint_loss'].update_state(joint_loss)
        self.train_trackers['scale'].update_state(scale)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass
        image_loss, joint_loss, scale, loss = self.compute_loss(y, y_pred, x)

        self.test_trackers['image_loss'].update_state(image_loss)
        self.test_trackers['joint_loss'].update_state(joint_loss)
        self.test_trackers['scale'].update_state(scale)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint = x
        y_image, y_joint = y

        image_loss = tf.reduce_mean(tf.square(y_image - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        roi = y_pred[2][0,-1]
        s = roi[2] - roi[0]
        scale = s
        loss = image_loss + joint_loss
        return image_loss, joint_loss, scale, loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_lstm(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='lstm'):
    roi_dim = 3
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof + roi_dim
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(state_dim)(x)

    predicted_ivec = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(x)
    predicted_jvec = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:image_vec_dim+dof], output_shape=(dof,))(x)
    roi_param = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim+dof:], output_shape=(roi_dim,))(x)
    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(roi_param) # (x, y, s)

    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])
    
    m = keras.Model([imgvec_input, joint_input], [predicted_ivec, predicted_jvec, roi], name=name)
    m.summary()
    return m

def model_roi_ae_lstm(input_image_shape, time_window_size, image_vec_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder(input_image_shape, time_window_size, image_vec_dim)(image_input)
    predicted_ivec, predicted_jvec, roi =  model_lstm(time_window_size, image_vec_dim, dof)([encoded_img, joint_input])

    decoded_roi = model_decoder(input_image_shape, image_vec_dim)(predicted_ivec)
    last_frame = tf.keras.layers.Lambda(lambda x:x[:,-1], output_shape=())(image_input)
    embedded_img = tf.keras.layers.Lambda(embed)([last_frame, decoded_roi, roi])

    m = AutoencoderWithCrop(inputs=[image_input, joint_input],
                            outputs=[embedded_img, predicted_jvec, roi],
                            name='roi_ae_lstm_embed')
    m.summary()
    return m

def conv1_block(x, out_channels):
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.MaxPool2D(pool_size=2)(x)

def model_encoder1(input_shape, out_dim, noise_stddev=0.2, name='encoder'):
    image_input = tf.keras.Input(shape=(input_shape,))

    # Denoising Autoencoder
    x = tf.keras.layers.GaussianNoise(noise_stddev)(image_input)

    x = conv1_block(x, 8)
    x = conv1_block(x, 16)
    x = conv1_block(x, 32)
    x = conv1_block(x, 64)

    x = tf.keras.layers.Flatten()(x)
    encoder_output = tf.keras.layers.Dense(out_dim, activation='selu')(x)

    encoder = keras.Model([image_input], encoder_output, name=name)
    encoder.summary()
    return encoder

def model_roi_decoder(output_shape, image_vec_dim, name='roi_decoder'):
    channels = output_shape[2]
    nblocks = 4
    h = int(output_shape[0]/2**nblocks)
    w = int(output_shape[1]/2**nblocks)

    imgvec_input = tf.keras.Input(shape=(image_vec_dim))
    roi_input = tf.keras.Input(shape=(3)) # x,y and scale

    # concatenate image vector and roi vector
    concat_vec = tf.keras.layers.concatenate([imgvec_input, roi_input])

    x = tf.keras.layers.Dense(h*w*channels, activation='selu')(concat_vec)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Reshape(target_shape=(h, w, channels))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = deconv_block(x, 64)
    x = deconv_block(x, 32)
    x = deconv_block(x, 16)
    x = deconv_block(x, 8)

    decoder_output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    decoder = keras.Model(imgvec_input, decoder_output, name=name)
    decoder.summary()
    return decoder

def model_roi_ae(input_shape, image_vec_dim, name='roi_ae'):
    image_input = tf.keras.Input(shape=(input_image_shape))
    roi_input = tf.keras.Input(shape=(2))

    # encode the whole image
    encoder = model_encoder1(input_shape, image_vec_dim)
    image_vec = encoder(image_input)

    # decode to the specifiec ROI image
    decoder = model_roi_decoder(output_shape, image_vec_dim)
    roi_image = decoder(image_vec, roi_input)

    # returns the whole model together with encoder and decoder for the separate use
    roi_ae = keras.Model([image_input, roi_input], [roi_image], name=name)

    return encoder, decoder, roi_ae

def model_roi_ae_train(input_shape, image_vec_dim, name='roi_ae_train'):
    image_input = tf.keras.Input(shape=(input_image_shape))
    e, d, ae = model_roi_ae(input_shape, image_vec_dim)

    roi_input = random()
    roi_image = ae(image_input, roi_input)
    m = keras.Model(image_input, [roi_image, roi_input], name=name)
    m.summary()
    return m



model = model_roi_ae_lstm(input_image_size+(3,), time_window_size, latent_dim, dof)

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

def prepare_for_test(cp='ae_cp.peg-in-hole-no-shadow.roi_ae_lstm_embed.20220214120204'):
    val_ds = Dataset(dataset)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

