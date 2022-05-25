# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np

def conv_block(x, out_channels):
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.MaxPool2D(pool_size=2)(x)

def time_distributed_conv_block(x, out_channels):
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    return tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

def model_encoder(input_shape, out_dim, noise_stddev=0.2, name='encoder'):
    image_input = tf.keras.Input(shape=(input_shape))

    x = tf.keras.layers.GaussianNoise(noise_stddev)(image_input) # Denoising Autoencoder

    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)

    x = tf.keras.layers.Flatten()(x)
    encoder_output = tf.keras.layers.Dense(out_dim, activation='selu')(x)

    encoder = keras.Model([image_input], encoder_output, name=name)
    encoder.summary()
    return encoder

def model_time_distributed_encoder(input_shape, time_window_size, out_dim, noise_stddev=0.2, name='encoder'):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_shape))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(noise_stddev))(image_input) # Denoising Autoencoder

    x = time_distributed_conv_block(x, 8)
    x = time_distributed_conv_block(x, 16)
    x = time_distributed_conv_block(x, 32)
    x = time_distributed_conv_block(x, 64)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    encoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dim, activation='selu'))(x)

    encoder = keras.Model([image_input], encoder_output, name=name)
    encoder.summary()
    return encoder

def model_lstm(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='lstm'):
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(state_dim)(x)
    imgvec_output = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(x)
    joint_output = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:], output_shape=(dof,))(x)

    lstm = keras.Model([imgvec_input, joint_input], [imgvec_output, joint_output], name=name)
    lstm.summary()
    return lstm

def deconv_block(x, out_channels):
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.UpSampling2D()(x)

def model_decoder(output_shape, image_vec_dim, name='decoder'):
    channels = output_shape[2]
    nblocks = 4
    h = int(output_shape[0]/2**nblocks)
    w = int(output_shape[1]/2**nblocks)

    imgvec_input = tf.keras.Input(shape=(image_vec_dim))
    x = tf.keras.layers.Dense(h*w*channels, activation='selu')(imgvec_input)
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

class AutoEncoderModel(tf.keras.Model):
    def __init__(self, *args, **kargs):
        super(AutoEncoderModel, self).__init__(*args, **kargs)
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        input_noise = tf.random.uniform(shape=(batch_size, 2), minval=-1, maxval=1)

        with tf.GradientTape() as tape:
            y_pred = self((x, input_noise), training=True) # Forward pass
            loss = self.compute_loss(y, y_pred, input_noise)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        input_noise = tf.zeros(shape=(batch_size, 2))

        y_pred = self((x, input_noise), training=False) # Forward pass
        loss = self.compute_loss(y, y_pred, input_noise)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y, y_pred, input_noise):
        y_aug = translate_image(y, input_noise)
        loss = tf.reduce_mean(tf.square(y_aug - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())

class AeLstmModel(tf.keras.Model):
    def __init__(self, *args, **kargs):
        super(AeLstmModel, self).__init__(*args, **kargs)
        tracker_names = ['image_loss', 'joint_loss','loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

        self.n_steps = 5

    def train_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.random.uniform(shape=(batch_size, 2), minval=-1, maxval=1)

        image_loss = 0.0
        joint_loss = 0.0
        y_images, y_joints = y
        y_images_tr = tf.transpose(y_images, [1,0,2,3,4])
        y_joints_tr = tf.transpose(y_joints, [1,0,2])

        with tf.GradientTape() as tape:
            for n in range(self.n_steps):
                pred_image, pred_joint = self((x_image, x_joint, input_noise), training=True) # Forward pass

                y_image_aug = translate_image(y_images_tr[n], input_noise)
                image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
                joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

                if n < self.n_steps:
                    x_image_tr = tf.transpose(x_image, [1,0,2,3,4])
                    x_image_tr_list = tf.unstack(x_image_tr)
                    x_image_tr_list[1:].append(pred_image)
                    x_image_tr = tf.stack(x_image_tr_list)
                    x_image = tf.transpose(x_image_tr, [1,0,2,3,4])
                    x_joint_tr = tf.transpose(x_joint, [1,0,2])
                    x_joint_tr_list = tf.unstack(x_joint_tr)
                    x_joint_tr_list[1:].append(pred_joint)
                    x_joint_tr = tf.stack(x_joint_tr_list)
                    x_joint = tf.transpose(x_joint_tr, [1,0,2])

            image_loss /= self.n_steps
            joint_loss /= self.n_steps
            loss = image_loss + joint_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['image_loss'].update_state(image_loss)
        self.train_trackers['joint_loss'].update_state(joint_loss)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.zeros(shape=(batch_size, 2))

        image_loss = 0.0
        joint_loss = 0.0
        y_images, y_joints = y

        y_images_tr = tf.transpose(y_images, [1,0,2,3,4])
        y_joints_tr = tf.transpose(y_joints, [1,0,2])

        for n in range(self.n_steps):
            pred_image, pred_joint = self((x_image, x_joint, input_noise), training=False) # Forward pass

            y_image_aug = translate_image(y_images_tr[n], input_noise)
            image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
            joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

            if n < self.n_steps:
                x_image_tr = tf.transpose(x_image, [1,0,2,3,4])
                x_image_tr_list = tf.unstack(x_image_tr)
                x_image_tr_list[1:].append(pred_image)
                x_image_tr = tf.stack(x_image_tr_list)
                x_image = tf.transpose(x_image_tr, [1,0,2,3,4])
                x_joint_tr = tf.transpose(x_joint, [1,0,2])
                x_joint_tr_list = tf.unstack(x_joint_tr)
                x_joint_tr_list[1:].append(pred_joint)
                x_joint_tr = tf.stack(x_joint_tr_list)
                x_joint = tf.transpose(x_joint_tr, [1,0,2])

        image_loss /= self.n_steps
        joint_loss /= self.n_steps
        loss = image_loss + joint_loss

        self.test_trackers['image_loss'].update_state(image_loss)
        self.test_trackers['joint_loss'].update_state(joint_loss)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y, y_pred, input_noise):
        y_image, y_joint = y
        # image_aug = translate_image(y_image, input_noise)
        y_image_aug = translate_image(y_image[:,0], input_noise[:,0])

        image_loss = tf.reduce_mean(tf.square(y_image_aug - y_pred[0]))
        # joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        joint_loss = tf.reduce_mean(tf.square(y_joint[:,0] - y_pred[1]))
        loss = image_loss + joint_loss
        return image_loss, joint_loss, loss

    def compute_loss2(self, y, y_pred, input_noise):
        y_images, y_joints = y
        y_pred_images, y_pred_joints = y_pred

         #image_aug = translate_image(y_image, input_noise)

        image_loss = tf.reduce_mean(tf.square(y_images - y_pred_images)) / self.n_steps
        joint_loss = tf.reduce_mean(tf.square(y_joints - y_pred_joints)) / self.n_steps
        loss = image_loss + joint_loss
        return image_loss, joint_loss, loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_autoencoder(input_image_shape, latent_dim, use_color_augmentation=False, use_geometrical_augmentation=True):
    image_input = tf.keras.Input(shape=(input_image_shape))
    input_noise = tf.keras.Input(shape=(2,))

    x = image_input

    if use_color_augmentation:
        x = ColorAugmentation()(x)
    if use_geometrical_augmentation:
        x = GeometricalAugmentation()(x, input_noise)

    encoded_img = model_encoder(input_image_shape, latent_dim)(x)
    decoded_img = model_decoder(input_image_shape, latent_dim)(encoded_img)

    if use_geometrical_augmentation:
        model = AutoEncoderModel(inputs=[image_input, input_noise], outputs=[decoded_img], name='autoencoder')
    else:
        model = Model(inputs=[image_input], outputs=[decoded_img], name='autoencoder')
    model.summary()

    return model

def model_ae_lstm(input_image_shape, time_window_size, latent_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_time_distributed_encoder(input_image_shape, time_window_size, latent_dim)(image_input)
    predicted_ivec, predicted_jvec = model_lstm(time_window_size, latent_dim, dof)([encoded_img, joint_input])
    decoded_img = model_decoder(input_image_shape, latent_dim)(predicted_ivec)

    model = tf.keras.Model(inputs=[image_input, joint_input],
                           outputs=[decoded_img, predicted_jvec],
                           name='ae_lstm')
    model.summary()

    return model

class ColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, brightness_max_delta=0.2,
                     contrast_lower=0.8, contrast_upper=1.2,
                     hue_max_delta=0.05):
        super().__init__()
        self.brightness_max_delta = brightness_max_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.hue_max_delta = hue_max_delta

    def call(self, inputs, training=None):
        return K.in_train_phase(self.augment_per_image(inputs), inputs, training=training)

    def augment_per_image(self, img):
        img = tf.image.random_brightness(img, max_delta=self.brightness_max_delta)
        img = tf.image.random_contrast(img, lower=self.contrast_lower, upper=self.contrast_upper)
        img = tf.image.random_hue(img, max_delta=self.hue_max_delta)
        return img

    def get_config(self):
        config = {
            "brightness_max_delta" : self.brightness_max_delta,
            "contrast_lower" : self.contrast_lower,
            "contrast_upper" : self.contrast_upper,
            "hue_max_delta" : self.hue_max_delta
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def translate_image(image, noise):
    # random_shift only works for array and eager tensor
    # img = tf.keras.preprocessing.image.random_shift(img, 0.02, 0.02,
    #                                                     row_axis=0, col_axis=1, channel_axis=2)
    # tf.image.crop_to_bounding_box, tf.image.pad_to_bounding_box
    shift = 4
    return tfa.image.translate(image, translations=shift*noise, fill_mode='constant')

class GeometricalAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, image, noise, training=None):
        return K.in_train_phase(self.augment_per_image(image, noise), image, training=training)

    def augment_per_image(self, img, noise):
        return translate_image(img, noise)

    def get_config(self):
        config = {
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedGeometricalAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, images, noise, training=None):
        return K.in_train_phase(tf.map_fn(self.augment_per_seq, (images, noise), fn_output_signature=tf.TensorSpec(shape=[20,80,160,3],dtype=tf.float32)),
                                    images, training=training)

    def augment_per_seq(self, args):
        return translate_image(args[0], args[1])

    def get_config(self):
        config = {
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CustomAugmentation(tf.keras.layers.Layer):
    def __init__(self, brightness_max_delta=0.2,
                     contrast_lower=0.8, contrast_upper=1.2,
                     hue_max_delta=0.05):
        super().__init__()
        self.brightness_max_delta = brightness_max_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.hue_max_delta = hue_max_delta

    def call(self, inputs, training=None):
        return K.in_train_phase(tf.map_fn(self.augment_per_image, inputs),
                                          inputs, training=training)

    def augment_per_image(self, img):
        img = tf.image.random_brightness(img, max_delta=self.brightness_max_delta)
        img = tf.image.random_contrast(img, lower=self.contrast_lower, upper=self.contrast_upper)
        img = tf.image.random_hue(img, max_delta=self.hue_max_delta)

        img = tfa.image.translate(img, translations=[80*0.05, 160*0.05], fill_mode='constant')
        return img

    def get_config(self):
        config = {
            "brightness_max_delta" : self.brightness_max_delta,
            "contrast_lower" : self.contrast_lower,
            "contrast_upper" : self.contrast_upper,
            "hue_max_delta" : self.hue_max_delta
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def model_ae_lstm_aug(input_image_shape, time_window_size, latent_dim, dof, joint_noise=0.03):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    input_noise = tf.keras.Input(shape=(2,))

    x = image_input
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomContrast(factor=0.2))(x)
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomBrightness(factor=0.2))(x)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomTranslation(height_factor=0.02, width_factor=0.02, fill_mode='constant', interpolation='bilinear', seed=None, fill_value=0.0))(x)

    x = TimeDistributedGeometricalAugmentation()(image_input, input_noise)

    encoded_img = model_time_distributed_encoder(input_image_shape, time_window_size, latent_dim)(x)

    joint_input_with_noise = tf.keras.layers.GaussianNoise(joint_noise)(joint_input)
    predicted_ivec, predicted_jvec = model_lstm(time_window_size, latent_dim, dof)([encoded_img, joint_input_with_noise])
    decoded_img = model_decoder(input_image_shape, latent_dim)(predicted_ivec)

    model = AeLstmModel(inputs=[image_input, joint_input, input_noise],
                            outputs=[decoded_img, predicted_jvec],
                            name='ae_lstm_aug')
    model.summary()
    return model
