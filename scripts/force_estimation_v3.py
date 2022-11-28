# -*- coding: utf-8 -*-

# from typing import Concatenate
import os, time
from datetime import datetime
from pybullet_tools import *
import force_distribution_viewer
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import res_unet
import tensorflow as tf
# import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from core.utils import *
from force_estimation_data_loader import ForceEstimationDataLoader
from model import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

dataset = 'basket-filling'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62
batch_size = 32

dl = ForceEstimationDataLoader(
                            os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset),
                            os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset+'-real'),
                            image_height=image_height,
                            image_width=image_width,
                            start_seq=1,
                            n_seqs=20,  # n_seqs=1500,
                            start_frame=3, n_frames=3)


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


def model_fmap_decoder(input_shape, name='fmap_decoder'):
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
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_image_decoder(input_shape, name='image_decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [1024, 512], 3, strides=[1, 1], name='resb1')
    x = res_unet.upsample(x, (24, 32))    
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (48, 64))
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb3')
    x = res_unet.upsample(x, (96, 128))
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')
    x = res_unet.upsample(x, (192, 256))
    x = res_unet.res_block(x, [16, 16], 3, strides=[1, 1], name='resb5')
    x = res_unet.upsample(x, (384, 512))
    x = res_unet.res_block(x, [8, 8], 3, strides=[1, 1], name='resb6')

    x = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(360, 512)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_rgb_to_fmap(input_shape=input_image_shape, input_noise_stddev=0.3):
    input_shape = input_shape + [3]
    image_input = tf.keras.Input(shape=input_shape)

    x = image_input

    # augmentation layers
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    resnet50 = ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)

    decoded_fmap = model_fmap_decoder((12, 16, 2048))(encoded_img)
    convnet = keras.Model(inputs=[image_input], outputs=[decoded_fmap], name='convnet')
    convnet.summary()

    decoded_img = model_image_decoder((12, 16, 2048))(encoded_img)
    convae = keras.Model(inputs=[image_input], outputs=[decoded_img], name='convae')
    convae.summary()

    return convnet, convae


task = 'train'
convnet, convae = model_rgb_to_fmap()
# model_file = 'ae_cp.basket-filling.model_resnet.20221125134301'

if task == 'train':
    train_data, valid_data = dl.load_data_for_rgb2fmap(train_mode=True)
#    trainer = Trainer(model, train_data, valid_data)
# elif task == 'adaptation':
#     model = model_rgb_to_fmap()
#     train_data, valid_data = dl.load_data_for_rgb2fmap_with_real(train_mode=True)
#     trainer = Trainer(model, train_data, valid_data)
# elif task == 'test':
#     model = model_rgb_to_fmap()
#     test_data = dl.load_data_for_rgb2fmap(test_mode=True)
#     tester = Tester(model, test_data, model_file)
# elif task == 'test-real':
#     model = model_rgb_to_fmap()
#     test_data = dl.load_real_data_for_rgb2fmap(test_mode=True)
#     tester = Tester(model, test_data, model_file)
# elif task == 'pick':
#     model = model_rgb_to_fmap()
#     test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
#     tester = Tester(model, test_data, model_file)


src_optimizer = keras.optimizers.Adamax(learning_rate=0.001)
tgt_optimizer = src_optimizer  # use the same optimizer for src and tgt domains

# convnet.compile(loss='mse', optimizer=src_optimizer)
# convae.compile(loss='mse', optimizer=tgt_optimizer)


def loss_fn(y_labels, y_prediction):
    loss = tf.reduce_mean(tf.square(y_labels - y_prediction))
    return loss


train_src_acc_metric = keras.metrics.Mean(name='src_acc')
val_src_acc_metric = keras.metrics.Mean(name='val_src_acc')
train_tgt_acc_metric = keras.metrics.Mean(name='tgt_acc')
val_tgt_acc_metric = keras.metrics.Mean(name='val_tgt_acc')


@tf.function
def train_step_X(X_batch, Y_batch):
    X_batch, Y_batch = augment(X_batch, Y_batch)

    with tf.GradientTape() as tape:
        Y_pred = convnet(X_batch, training=True)
        loss = loss_fn(Y_batch, Y_pred)
    gradients = tape.gradient(loss, convnet.trainable_weights)
    src_optimizer.apply_gradients(zip(gradients, convnet.trainable_weights))
    train_src_acc_metric.update_state(loss)
    return loss


@tf.function
def train_step_Xu(Xu_batch):
    with tf.GradientTape() as tape:
        Yu_pred = convae(Xu_batch, training=True)
        loss = loss_fn(Xu_batch, Yu_pred)
    gradients = tape.gradient(loss, convae.trainable_weights)
    src_optimizer.apply_gradients(zip(gradients, convae.trainable_weights))
    train_tgt_acc_metric.update_state(loss)
    return loss


@tf.function
def test_step_X(X_batch, Y_batch):
    Y_pred = convnet(X_batch, training=False)
    loss = loss_fn(Y_batch, Y_pred)
    val_src_acc_metric.update_state(loss)


@tf.function
def test_step_Xu(Xu_batch):
    Y_pred = convae(Xu_batch, training=False)
    loss = loss_fn(Xu_batch, Y_pred)
    val_tgt_acc_metric.update_state(loss)


def train_drcn(X, Y, Xu, val_X, val_Y, val_Xu, 
               batch_size, epochs=10,
               early_stop_patience=100,
               reduce_lr_patience=2):
    """
    Args:
        X (np.array) : array of source images
        Y (np.array) : array of source labels (force maps)
        Xu (np.array) : array of target images

    """
    X = X.astype('float32')
    Y = Y.astype('float32')
    Xu = Xu.astype('float32')
    val_X = val_X.astype('float32')
    val_Y = val_Y.astype('float32')
    val_Xu = val_Xu.astype('float32')

    X_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(X)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    Xu_dataset = tf.data.Dataset.from_tensor_slices(Xu).shuffle(len(Xu)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_X_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_Y)).shuffle(len(val_X)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_Xu_dataset = tf.data.Dataset.from_tensor_slices(val_Xu).shuffle(len(val_Xu)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    d = os.path.join(os.path.dirname(os.getcwd()), 'runs')
    f = '{}.{}.{}'.format(dataset, 'rgb2force_drcn', datetime.now().strftime('%Y%m%d%H%M%S'))
    convnet_cp_path = os.path.join(d, f, 'convnet.ckpt')
    convae_cp_path = os.path.join(d, f, 'convae.ckpt')

    best_acc = 1e+10
    best_epoch = 0
    reduce_lr_epoch = 0
    for epoch in range(epochs):
        print('\nStart epoch', epoch)
        start_time = time.time()

        for step, tgt_data in enumerate(Xu_dataset):
            convae_loss = train_step_Xu(tgt_data)
            # if step % 200 == 0:
            #     print('target loss at step %d: %.2f' % (step, float(convae_loss)))

        for step, src_data in enumerate(X_dataset):
            convnet_loss = train_step_X(*src_data)
            # if step % 200 == 0:
            #     print('source loss at step %d: %.2f' % (step, float(convnet_loss)))

        # Display metrics at the end of each epoch.
        train_tgt_acc = train_tgt_acc_metric.result()
        train_src_acc = train_src_acc_metric.result()
        print('Training loss: target=%.4f, source=%.4f / ' % (float(train_tgt_acc), float(train_src_acc)), end='')

        # Reset training metrics at the end of each epoch.
        train_tgt_acc_metric.reset_states()
        train_src_acc_metric.reset_states()

        for val_tgt_data in val_Xu_dataset:
            test_step_Xu(val_tgt_data)

        for val_src_data in val_X_dataset:
            test_step_X(*val_src_data)

        val_tgt_acc = val_tgt_acc_metric.result()
        val_src_acc = val_src_acc_metric.result()
        val_tgt_acc_metric.reset_states()
        val_src_acc_metric.reset_states()
        print('Validation loss: target=%.4f, source=%.4f' % (float(val_tgt_acc), float(val_src_acc)))

        if float(val_src_acc) < best_acc:
            best_acc = float(val_src_acc)
            best_epoch = epoch
            convnet.save_weights(convnet_cp_path, save_format='tf')
            convae.save_weights(convae_cp_path, save_format='tf')
            print('saving weights to  %s and %s' % (convnet_cp_path, convae_cp_path))

        if epoch - best_epoch > reduce_lr_patience and epoch - reduce_lr_epoch > reduce_lr_patience:
            new_lr = src_optimizer.lr * 0.1
            if new_lr > 1e-5:
                src_optimizer.lr = new_lr
                tgt_optimizer.lr = new_lr
                reduce_lr_epoch = epoch
                print('reduce learning rate to %.6f' % new_lr)

        if epoch - best_epoch > early_stop_patience:
            print('early stop')
            break

        print('Time taken: %.2fs' % (time.time() - start_time))


class Trainer:
    def __init__(self, model,
                 train_dataset,
                 val_dataset,
                 # load_weight=False,
                 batch_size=32,
                 runs_directory=None,
                 checkpoint_file=None):

        self.input_image_shape = val_dataset[0][0].shape
        self.batch_size = batch_size

        self.model = model
        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        if train_dataset:
            self.train_ds = train_dataset

        if val_dataset:
            self.val_ds = val_dataset
            self.val_data_loaded = True

        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.getcwd()), 'runs')
        f = checkpoint_file if checkpoint_file else 'ae_cp.{}.{}.{}'.format(dataset, self.model.name, datetime.now().strftime('%Y%m%d%H%M%S'))
        self.checkpoint_path = os.path.join(d, f, 'cp.ckpt')

        if checkpoint_file:
            print('load weights from ', checkpoint_file)
            self.model.load_weights(self.checkpoint_path)

    def prepare_callbacks(self, early_stop_patience=100, reduce_lr_patience=50):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         mode='min',
                                                         save_best_only=True)

        # early stopping if not changing for 50 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=early_stop_patience)

        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=reduce_lr_patience,
                                                         verbose=1,
                                                         min_lr=0.00001)

        profiler = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                  histogram_freq=1,
                                                  profile_batch='15,25')
        return cp_callback, early_stop, reduce_lr, profiler

    def train(self, epochs=300, early_stop_patience=100, reduce_lr_patience=50):
        xs = self.train_ds[0].astype('float32')
        ys = self.train_ds[1].astype('float32')
        val_xs = self.val_ds[0].astype('float32')
        val_ys = self.val_ds[1].astype('float32')

        start = time.time()
        callbacks = self.prepare_callbacks(early_stop_patience, reduce_lr_patience)

        source_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(xs)).batch(self.batch_size)
        val_source_dataset = tf.data.Dataset.from_tensor_slices((val_xs, val_ys)).batch(self.batch_size)

        history = self.model.fit(source_dataset,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=val_source_dataset,
                                 shuffle=True)

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))


def visualize_forcemaps(force_distribution, title='', zaxis_first=False):
    f = force_distribution / np.max(force_distribution)
    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(title, fontsize=28)

    if zaxis_first:
        channels = f.shape[0]
    else:
        channels = f.shape[-1]
    for p in range(channels):
        ax = fig.add_subplot(channels//10, 10, p+1)
        ax.axis('off')
        if zaxis_first:
            ax.imshow(f[p], cmap='gray', vmin=0, vmax=1.0)
        else:
            ax.imshow(f[:, :, p], cmap='gray', vmin=0, vmax=1.0)


def visualize_result(f_prediction, f_label, rgb, filename=None):
    visualize_forcemaps(f_prediction, title='prediction')
    plt.savefig('prediction.png')
    visualize_forcemaps(f_label, title='ground truth')
    plt.savefig('ground_truth.png')
    p = plt.imread('prediction.png')[:, :, :3]
    g = plt.imread('ground_truth.png')[:, :, :3]
    pg = np.concatenate([p, g], axis=0)
    rgb2 = np.ones((800, 512, 3))
    rgb2[220:580] = rgb
    res = np.concatenate([rgb2, pg], axis=1)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(res)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


class Tester:
    def __init__(self, model, test_data, checkpoint_file, runs_directory=None):
        self.model = model
        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.getcwd()), 'runs')
        self.checkpoint_path = os.path.join(d, checkpoint_file, 'cp.ckpt')
        print('load weights from ', self.checkpoint_path)
        self.model.load_weights(self.checkpoint_path)
        self.test_data = test_data

    def predict(self, n):
        xs = self.test_data[0][n:n+1]
        ys = self.test_data[1][n:n+1]
        y_pred = self.model.predict(xs)
        plt.imshow(xs[0])
        visualize_forcemaps(y_pred[0])
        visualize_forcemaps(ys[0])
        plt.show()

    def predict_segmentation_mask(self, n):
        xs = self.test_data[0][n:n+1]
        ys = self.test_data[1][n:n+1]
        y_pred_logits = self.model.predict(xs)
        plt.figure()
        plt.imshow(xs[0])
        y_pred = tf.cast(tf.argmax(y_pred_logits, axis=-1), dtype=tf.uint8)
        plt.figure()
        plt.imshow(y_pred[0])
        plt.figure()
        plt.imshow(ys[0])
        plt.show()
        return xs[0], ys[0], y_pred[0]

    def predict_force_from_rgb(self, n, visualize=True):
        xs = self.test_data[0][n:n+1]
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        force_label = self.test_data[1][n]
        if visualize_result:
            visualize_result(y_pred_forces[0], force_label, xs[0], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, xs[0]

    def predict_force_from_rgb_with_img(self, rgb):
        xs = np.expand_dims(rgb, axis=0)
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        visualize_forcemaps(y_pred_forces[0])
        plt.show()
        return y_pred_forces

    def predict_force_from_depth_seg(self, n, rgb):
        xs_depth = self.test_data[0][0][n:n+1]
        xs_seg = self.test_data[0][1][n:n+1]
        xs = xs_depth, xs_seg
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        force_label = self.test_data[1][n]

        visualize_result(y_pred_forces[0], force_label, rgb[n], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, rgb[n]


X, Y = train_data
val_X, val_Y = valid_data
Xu, val_Xu = dl.load_real_data_for_rgb2fmap(train_mode=True)
train_drcn(X, Y, Xu, val_X, val_Y, val_Xu, batch_size, epochs=20)
