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


dataset = 'reaching'
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
    es = 0.15 * (1.0 + s)
    img_center = tf.map_fn(fn=lambda x: tf.constant([0.5, 0.5]), elems=c)
    a = tf.tile(tf.expand_dims(es, 1), (1,2))
    lt = img_center + 0.4 * (c - img_center) - a
    rb = img_center + 0.4 * (c - img_center) + a
    roi = tf.concat([lt, rb], axis=1)
    return roi

class CustomModel(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(CustomModel, self).__init__(*args, **kargs)
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.joint_loss_tracker = keras.metrics.Mean(name="joint_loss")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_image_loss_tracker = keras.metrics.Mean(name="val_image_loss")
        self.val_joint_loss_tracker = keras.metrics.Mean(name="val_joint_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            image_loss, joint_loss, loss = self.compute_loss(y, y_pred, x)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.image_loss_tracker.update_state(image_loss)
        self.joint_loss_tracker.update_state(joint_loss)
        self.loss_tracker.update_state(loss)

        return {
            "image_loss": self.image_loss_tracker.result(),
            "joint_loss": self.joint_loss_tracker.result(),
            "loss": self.loss_tracker.result(),
            }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass
        val_image_loss, val_joint_loss, val_loss = self.compute_loss(y, y_pred, x)

        self.val_image_loss_tracker.update_state(val_image_loss)
        self.val_joint_loss_tracker.update_state(val_joint_loss)
        self.val_loss_tracker.update_state(val_loss)

        return {
            "image_loss": self.val_image_loss_tracker.result(),
            "joint_loss": self.val_joint_loss_tracker.result(),
            "loss": self.val_loss_tracker.result(),
            }

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint = x
        y_image, y_joint = y

        image_loss = tf.reduce_mean(tf.square(y_image - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        loss = image_loss + joint_loss
        return image_loss, joint_loss, loss

    @property
    def metrics(self):
        return [
            self.image_loss_tracker,
            self.joint_loss_tracker,
            self.loss_tracker,
            self.val_image_loss_tracker,
            self.val_joint_loss_tracker,
            self.val_loss_tracker,
            ]

def model_lstm_no_split(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='lstm'):
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(state_dim)(x)

    lstm = keras.Model([imgvec_input, joint_input], x, name=name)
    lstm.summary()
    return lstm

def model_roi_mlp(latent_dim, name='roi_mlp'):
    imgvec_input = tf.keras.Input(shape=(latent_dim))

    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(imgvec_input) # (x, y, s)
    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])

    m = keras.Model(imgvec_input, roi, name=name)
    m.summary()
    return m

def conv1_block(x, out_channels):
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.MaxPool2D(pool_size=2)(x)

def model_encoder_decoder(input_shape, noise_stddev=0.2, name='encoder_decoder'):
    image_input = tf.keras.Input(shape=(input_shape))

    x = tf.keras.layers.GaussianNoise(noise_stddev)(image_input)

    x = conv1_block(x, 8)
    x = conv1_block(x, 16)
    x = conv1_block(x, 32)
    x = conv1_block(x, 64)
    x = deconv_block(x, 32)
    x = deconv_block(x, 16)
    x = deconv_block(x, 8)
    x = deconv_block(x, 3)

    m = keras.Model([image_input], x, name=name)
    m.summary()
    return m

def model_weighted_roi_loss(input_image_shape, time_window_size, latent_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder(input_image_shape, time_window_size, latent_dim)(image_input)
    predicted = model_lstm_no_split(time_window_size, latent_dim, dof)([encoded_img, joint_input])

    roi = model_roi_mlp(latent_dim+dof)(predicted)

    imgvec_pred = tf.keras.layers.Lambda(lambda x:x[:,:latent_dim], output_shape=(latent_dim,))(predicted)
    joint_pred = tf.keras.layers.Lambda(lambda x:x[:,latent_dim:], output_shape=(dof,))(predicted)
    decoded_pred = model_decoder(input_image_shape, latent_dim)(imgvec_pred)

    cropped_pred = tf.keras.layers.Lambda(crop_and_resize, output_shape=input_image_shape) ([decoded_pred, roi])
    predicted2 = model_encoder_decoder(input_image_shape)(cropped_pred)

    m = CustomModel(inputs=[image_input, joint_input],
                        outputs=[predicted2, joint_pred, roi],
                        name='time_space_prediction')
    m.summary()
    return m


model = model_weighted_roi_loss(input_image_size+(3,), time_window_size, latent_dim, dof)

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

def prepare_for_test(cp):
    val_ds = Dataset(dataset)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

