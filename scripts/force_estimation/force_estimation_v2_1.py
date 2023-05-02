# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from core import res_unet
from core.model import *
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model


image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62


def augment(xs, ys):
    img = xs
    fmap = ys

    # color transformation on xs
    # brightness_max_delta=0.2
    # contrast_lower=0.8
    # contrast_upper=1.2
    hue_max_delta = 0.05
    # img = tf.image.random_brightness(img, max_delta=brightness_max_delta)
    # img = tf.image.random_contrast(img, lower=contrast_lower, upper=contrast_upper)
    img = tf.image.random_hue(img, max_delta=hue_max_delta)

    # apply save transform to xs and ys
    batch_sz = tf.shape(xs)[0]
    height = tf.shape(xs)[1]
    # width = tf.shape(xs)[2]
    shift_fmap = 2.0
    shift_height = tf.cast(height * 4 / 40, tf.float32)
    shift_width = tf.cast(height * 4 / 40, tf.float32)
    angle_factor = 0.2
    rnds = tf.random.uniform(shape=(batch_sz, 2))
    img = tfa.image.translate(img, translations=tf.stack([shift_height, shift_width], axis=0)*rnds)
    fmap = tfa.image.translate(fmap, translations=tf.stack([shift_fmap, shift_fmap], axis=0)*rnds)
    rnds = tf.random.uniform(shape=(batch_sz,))
    rnds = rnds - 0.5
    img = tfa.image.rotate(img, angles=angle_factor*rnds)
    fmap = tfa.image.rotate(fmap, angles=angle_factor*rnds)

    # add gaussian noise to xs

    return img, fmap


class ForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(ForceEstimationModel, self).__init__(*args, **kargs)
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        xs, y_labels = data
        xs, y_labels = augment(xs, y_labels)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True)
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        xs, y_labels = data

        y_pred = self(xs, training=False)
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y_labels, y_pred):
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


class DRCNForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(DRCNForceEstimationModel, self).__init__(*args, **kargs)
        # tracker_names = ['loss', 'dloss', 'sloss', 'floss']
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        src_data, tgt_data = data
        xs, y_labels,  = src_data
        xs, y_labels = augment(xs, y_labels)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True)
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        src_data, tgt_data = data
        xs, y_labels,  = src_data

        y_pred = self(xs, training=False)
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y_labels, y_pred):
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_rgb_to_fmap(input_shape=input_image_shape,
                      num_filters=[16, 32, 64],
                      kernel_size=3,
                      num_channels=3,
                      num_classes=62,
                      noise_stddev=0.3):
    input = tf.keras.Input(shape=input_shape + [num_channels])

    # Data Augmentation
    x = tf.keras.layers.GaussianNoise(noise_stddev)(input)

    encoder_output = res_unet.encoder(x, num_filters, kernel_size)

    # bridge layer, number of filters is double that of the last encoder layer
    bridge = res_unet.res_block(encoder_output[-1], [num_filters[-1]*2], kernel_size, strides=[2, 1], name='bridge')

    print(encoder_output[-1].shape)
    # decoder_output = res_unet.decoder(bridge, encoder_output, num_filters, kernel_size)
    out1, out2, out3 = res_unet.multi_head_decoder(bridge, encoder_output, num_filters, kernel_size, num_heads=3)

    output_depth = tf.keras.layers.Conv2D(1, 
                                          kernel_size,
                                          strides=1,
                                          padding='same',
                                          name='depth_output')(out1)

    output_seg = tf.keras.layers.Conv2D(num_classes,
                                        kernel_size,
                                        strides=1,
                                        padding='same',
                                        name='seg_output')(out2)

    output_force = tf.keras.layers.Conv2D(20,
                                          kernel_size,
                                          strides=1,
                                          padding='same',
                                          name='force_output')(out3)
    output_force = tf.keras.layers.Resizing(40, 40)(output_force)

    model = ForceEstimationModel(inputs=[input],
                                 outputs=[output_depth, output_seg, output_force],
                                 name='force_estimator')

    model.summary()
    return model


def model_simple_decoder(input_shape, name='decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))

    x = feature_input
    x = deconv_block(x, 1024, with_upsampling=False)  # 12x16
    x = deconv_block(x, 512)                          # 24x32
    x = deconv_block(x, 256, with_upsampling=False)   # 24x32
    x = deconv_block(x, 128, with_upsampling=False)   # 24x32
    x = deconv_block(x, 64)                           # 48x64
    x = deconv_block(x, 32, with_upsampling=False)    # 48x64
    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)

    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_resnet_decoder(input_shape, name='resnet_decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [1024, 512], 3, strides=[1, 1], name='resb1')
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (24, 32))
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb3')
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = tf.keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_resnet_decoder_wide(input_shape, name='resnet_decoder_wide'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [1024, 512], 3, strides=[1, 1], name='resb1')
    x = res_unet.upsample(x, (24, 32))
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (48, 64))
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb3')
    x = res_unet.upsample(x, (96, 128))
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(30, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(120, 160)(x)
    decoder_output = x
    decoder = tf.keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_resnet_decoder2(input_shape, name='resnet_decoder2'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb1')
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (24, 32))  # (46, 64)
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_rgb_to_fmap_res50(input_shape=input_image_shape, input_noise_stddev=0.3):
    input_shape = input_shape + [3]
    image_input = tf.keras.Input(shape=input_shape)

    x = image_input

    # augmentation layers
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    # encoder
    resnet50 = ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)
    
    # decoder
    decoded_img = model_resnet_decoder((12, 16, 2048))(encoded_img)

    model = ForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    # model = DRCNForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    model.summary()

    return model


def model_rgb_to_fmap_res50_wide(input_shape=input_image_shape, input_noise_stddev=0.3):
    input_shape = input_shape + [3]
    image_input = tf.keras.Input(shape=input_shape)

    x = image_input

    # augmentation layers
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    # encoder
    resnet50 = ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)
    
    # decoder
    decoded_img = model_resnet_decoder_wide((12, 16, 2048))(encoded_img)

    model = ForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet_wide')
    model.summary()

    return model


def model_depth_to_fmap(input_shape=input_image_shape, kernel_size=3, num_classes=num_classes):
    input_depth = tf.keras.Input(shape=input_shape + [1])
    input_seg = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=num_classes))(input_seg)
    x = tf.keras.layers.Concatenate(axis=-1)([input_depth, x])

    encoder_output = res_unet.encoder(x, num_filters=(32, 64, 64, 128, 256), kernel_size=kernel_size)
    decoded_img = model_resnet_decoder2((23, 32, 256))(encoder_output[-1])

    model = Model(inputs=[input_depth, input_seg], outputs=[decoded_img], name='model_depth_to_fmap')
    model.summary()

    return model
