# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np


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
