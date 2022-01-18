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
    #es = 0.075 * (1.0 + s)
    #img_center = tf.tile(tf.constant([[0.5, 0.5]], dtype=tf.float32), (batch_size,1))
    img_center = tf.map_fn(fn=lambda x: tf.constant([0.5, 0.5]), elems=c)
    a = tf.tile(tf.expand_dims(es, 1), (1,2))
    lt = img_center + 0.4 * (c - img_center) - a
    rb = img_center + 0.4 * (c - img_center) + a
    roi = tf.concat([lt, rb], axis=1)
    roi3 = tf.expand_dims(roi, 0)
    return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

# def roi_rect(args):
#     """
#     Fixed ROI (averaged over estimated ROIs)
#     """
#     c, s = args
#     lt = tf.tile(tf.constant([[0.1205094, 0.15675367]], dtype=tf.float32), (batch_size,1))
#     rb = tf.tile(tf.constant([[0.70754665, 0.74379086]], dtype=tf.float32), (batch_size,1))
#     roi = tf.concat([lt, rb], axis=1)
#     roi3 = tf.expand_dims(roi, 0)
#     return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

# def roi_rect(args):
#     """
#     To invalidate ROI, use this function.
#     This returns the whole image as ROI
#     """
#     c, s = args
#     lt = tf.map_fn(fn=lambda x: tf.constant([0.0, 0.0]), elems=c)
#     rb = tf.map_fn(fn=lambda x: tf.constant([1.0, 1.0]), elems=c)
#     roi = tf.concat([lt, rb], axis=1)
#     roi3 = tf.expand_dims(roi, 0)
#     return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])


class CustomModel(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(CustomModel, self).__init__(*args, **kargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            loss = self.compute_loss(y, y_pred, x)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass
        val_loss = self.compute_loss(y, y_pred, x)

        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint = x
        y_image, y_joint = y

        y_pred_cropped = crop_and_resize((y_pred[0], roi[:,-1]))
        y_cropped = crop_and_resize((y_image, roi[:,-1]))

        roi_loss = tf.reduce_mean(tf.square(y_cropped - y_pred_cropped))
        image_loss = tf.reduce_mean(tf.square(y_image - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        loss = image_loss + joint_loss
        loss = l / (1+l) * image_loss + 1 / (1+l) * image_loss + joint_loss
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

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
    lstm.summary()
    return lstm

def model_roi_mlp(image_vec, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='roi_mlp'):

    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(image_vec) # (x, y, s)
    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])

    m = keras.Model([imgvec_input, joint_input], roi, name=name)
    m.summary()
    return m

def model_weighted_roi_loss(input_image_shape, time_window_size, latent_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder(input_image_shape, time_window_size, latent_dim)(image_input)
    predicted = model_lstm_no_split(time_window_size, latent_dim, dof)([encoded_image, joint_input])

    roi = model_roi_mlp()(predicted)

    imgvec_pred = tf.keras.layers.Lambda(lambda x:x[:,:latent_dim], output_shape=(latent_dim,))(predicted)
    joint_pred = tf.keras.layers.Lambda(lambda x:x[:,latent_dim:], output_shape=(dof,))(predicted)
    decoded_pred = model_decoder(input_image_shape, latent_dim)(imgvec_pred)

    m = CustomModel(inputs=[image_input, joint_input],
                        outputs=[decoded_pred, joint_pred, roi],
                        name='weighted_roi_loss')
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

