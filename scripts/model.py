# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def conv_block(x, out_channels):
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    return tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

def model_encoder(input_shape, time_window_size, out_dim, noise_stddev=0.2, name='encoder'):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_shape))

    # Denoising Autoencoder
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(noise_stddev))(image_input)

    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)

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

def model_ae_lstm(input_image_shape, time_window_size, latent_dim, dof):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder(input_image_shape, time_window_size, latent_dim)(image_input)
    predicted_ivec, predicted_jvec = model_lstm(time_window_size, latent_dim, dof)([encoded_img, joint_input])
    decoded_img = model_decoder(input_image_shape, latent_dim)(predicted_ivec)

    model = tf.keras.Model(inputs=[image_input, joint_input],
                           outputs=[decoded_img, predicted_jvec],
                           name='ae_lstm')
    model.summary()
    return model


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
        # random_shift only works for array and eager tensor
        # img = tf.keras.preprocessing.image.random_shift(img, 0.02, 0.02,
        #                                                     row_axis=0, col_axis=1, channel_axis=2)
        # tf.image.crop_to_bounding_box, tf.image.pad_to_bounding_box
        
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

    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomContrast(factor=0.2))(image_input)
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomBrightness(factor=0.2))(x)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.RandomTranslation(height_factor=0.02, width_factor=0.02, fill_mode='constant', interpolation='bilinear', seed=None, fill_value=0.0))(x)

    x = CustomAugmentation()(image_input)

    encoded_img = model_encoder(input_image_shape, time_window_size, latent_dim)(x)

    joint_input_with_noise = tf.keras.layers.GaussianNoise(joint_noise)(joint_input)
    predicted_ivec, predicted_jvec = model_lstm(time_window_size, latent_dim, dof)([encoded_img, joint_input_with_noise])
    decoded_img = model_decoder(input_image_shape, latent_dim)(predicted_ivec)

    model = tf.keras.Model(inputs=[image_input, joint_input],
                           outputs=[decoded_img, predicted_jvec],
                           name='ae_lstm')
    model.summary()
    return model
