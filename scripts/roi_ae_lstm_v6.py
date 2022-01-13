# -*- coding: utf-8 -*-

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from utils import *
from model import *
import trainer

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

# def roi_rect(args):
#     c, s = args
#     es = 0.15 * (1.0 + s)
#     #es = 0.075 * (1.0 + s)
#     #img_center = tf.tile(tf.constant([[0.5, 0.5]], dtype=tf.float32), (batch_size,1))
#     img_center = tf.map_fn(fn=lambda x: tf.constant([0.5, 0.5]), elems=c)
#     a = tf.tile(tf.expand_dims(es, 1), (1,2))
#     lt = img_center + 0.4 * (c - img_center) - a
#     rb = img_center + 0.4 * (c - img_center) + a
#     roi = tf.concat([lt, rb], axis=1)
#     roi3 = tf.expand_dims(roi, 0)
#     return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

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

def roi_rect(args):
    """
    To invalidate ROI, use this function.
    This returns the whole image as ROI
    """
    c, s = args
    lt = tf.map_fn(fn=lambda x: tf.constant([0.0, 0.0]), elems=c)
    rb = tf.map_fn(fn=lambda x: tf.constant([1.0, 1.0]), elems=c)
    roi = tf.concat([lt, rb], axis=1)
    roi3 = tf.expand_dims(roi, 0)
    return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])


class AutoencoderWithCrop(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(AutoencoderWithCrop, self).__init__(*args, **kargs)
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

        roi = y_pred[2]
        print(' ROI: ', roi)

        val_loss = self.compute_loss(y, y_pred, x)

        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint = x
        y_image, y_joint = y
        roi = y_pred[2]
        y_image_cropped = crop_and_resize((y_image, roi[:,-1]))

        # batch_size = x_image.shape[0]
        # reconst_roi = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), [batch_size, 1])
        # y_image_cropped = crop_and_resize((y_image, reconst_roi))

        image_loss = tf.reduce_mean(tf.square(y_image_cropped - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        loss = image_loss + joint_loss
        #loss = image_loss + 10.*joint_loss
        #loss = keras.losses.mean_squared_error(y_image_cropped, y_pred[0])
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]


def model_roi_lstm(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='roi_lstm'):
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)

    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(x) # (x, y, s)
    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])

    m = keras.Model([imgvec_input, joint_input], roi, name=name)
    m.summary()
    return m

def model_roi_encoder(input_shape, time_window_size, out_dim, noise_stddev=0.2, name='roi_encoder'):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_shape))
    roi_input = tf.keras.Input(shape=(time_window_size, 4))

    roi = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(crop_and_resize, output_shape=input_shape)) ([image_input, roi_input])
    roi_extractor = tf.keras.Model([image_input, roi_input], roi, name='roi_extractor')
    roi_extractor.summary()

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(noise_stddev))(roi)

    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    encoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dim, activation='selu'))(x)

    encoder = keras.Model([image_input, roi_input], encoder_output, name=name)
    encoder.summary()
    return encoder

def model_roi_ae_lstm(input_image_shape, time_window_size, latent_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder(input_image_shape, time_window_size, latent_dim)(image_input)
    roi = model_roi_lstm(time_window_size, latent_dim, dof)([encoded_img, joint_input])

    encoded_roi = model_roi_encoder(input_image_shape, time_window_size, latent_dim)([image_input, roi])
    predicted_ivec, predicted_jvec = model_lstm(time_window_size, latent_dim, dof)([encoded_roi, joint_input])
    decoded_roi = model_decoder(input_image_shape, latent_dim)(predicted_ivec)
    m = AutoencoderWithCrop(inputs=[image_input, joint_input],
                            outputs=[decoded_roi, predicted_jvec, roi],
                            name='roi_ae_lstm')
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

def prepare_for_test(cp='ae_cp.reaching.ae_lstm.20220112181145'):
    val_ds = Dataset(dataset)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.Trainer(model, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

