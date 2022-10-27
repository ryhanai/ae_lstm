# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2, os, time
from datetime import datetime
from model import *
from core.utils import *

dataset = 'basket-filling'

def load1(seqNo, frameNo):
    dataset_path = os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset)
    rgb_path = os.path.join(dataset_path, str(seqNo), 'rgb{:05d}.jpg'.format(frameNo))
    depth_path = os.path.join(dataset_path, str(seqNo), 'depth_zip{:05d}.pkl'.format(frameNo))
    seg_path = os.path.join(dataset_path, str(seqNo), 'seg{:05d}.png'.format(frameNo))            
    force_path = os.path.join(dataset_path, str(seqNo), 'force_zip{:05d}.pkl'.format(frameNo))
    bin_state_path = os.path.join(dataset_path, str(seqNo), 'bin_state{:05d}.pkl'.format(frameNo))
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = pd.read_pickle(depth_path, compression='zip')
    seg = plt.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    force = pd.read_pickle(force_path, compression='zip')
    bin_state = pd.read_pickle(bin_state_path)
    crop = 120
    rgb = rgb[:,crop:-crop] # crop the right & left side
    depth = depth[:,crop:-crop]
    seg = seg[:,crop:-crop]
    force = force[:,:,:20]
    return rgb, depth, seg, force, bin_state

def load_data():
    n_seqs = 100
    n_frames = 6
    x, y = np.mgrid[1:1+n_seqs, 0:n_frames]
    data_ids = list(zip(x.ravel(), y.ravel()))
    # data_ids = np.random.permutation(data_ids)
    # train
    X_train = np.empty((n_seqs*n_frames, 360, 512, 3))
    Y_train = np.empty((n_seqs*n_frames, 20, 40, 40))
    train_states = []
    
    # for i in range(0,100):
    #     print(i)
    #     seqNo, frameNo = data_ids[i]
    #     rgb, depth, seg, force, bs = load1(seqNo, frameNo)
    #     X_train[i] = rgb
    #     Y_train[i] = force
    #     train_states.append(bs)
    # # valid
    # X_valid = np.empty((250,480,480,3))
    # Y_valid = np.empty((250,40,40,40))
    # valid_states = []
    # for i in range(0,250):
    #     print(i)
    #     seqNo, frameNo = data_ids[1500+i]
    #     rgb, force, bs = load1(seqNo, frameNo)
    #     X_valid[i] = rgb
    #     Y_valid[i] = force
    #     valid_states.append(bs)
    # # test
    # X_test = np.empty((250,480,480,3))
    # Y_test = np.empty((250,40,40,40))
    # test_states = []
    # for i in range(0,250):
    #     print(i)
    #     seqNo, frameNo = data_ids[1750+i]
    #     rgb, force, bs = load1(seqNo, frameNo)
    #     X_test[i] = rgb
    #     Y_test[i] = force
    #     test_states.append(bs)
    # X_train /= 255.
    # X_valid /= 255.
    # X_test /= 255.
    # #Y_train = np.log(Y_train+1)
    # #Y_valid = np.log(Y_valid+1)
    # #Y_test = np.log(Y_test+1)
    # fmax = np.max(Y_train)
    # Y_train /= fmax
    # Y_valid /= fmax
    # Y_test /= fmax
    # Y_train = np.transpose(np.flip(Y_train, 2), (0,2,1,3))
    # Y_valid = np.transpose(np.flip(Y_valid, 2), (0,2,1,3))
    # Y_test = np.transpose(np.flip(Y_test, 2), (0,2,1,3))

    # return (X_train,Y_train,train_states), (X_valid,Y_valid,valid_states), (X_test,Y_test,test_states)

def model_encoder(input_shape, noise_stddev=0.2, name='encoder'):
    image_input = tf.keras.Input(shape=(input_shape))

    x = tf.keras.layers.GaussianNoise(noise_stddev)(image_input) # Denoising Autoencoder
    x = conv_block(x, 8) # 240
    x = conv_block(x, 16) # 120
    x = conv_block(x, 32) # 60
    x = conv_block(x, 64) # 30
    x = conv_block(x, 128) # 15

    encoder_output = x
    encoder = keras.Model([image_input], encoder_output, name=name)
    encoder.summary()
    return encoder

def model_decoder(input_shape, name='decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))

    x = feature_input
    x = deconv_block(x, 64, with_upsampling=False) # 15
    x = deconv_block(x, 64) # 30
    x = deconv_block(x, 40) # 60
    x = tf.keras.layers.Conv2DTranspose(40, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder

def model_conv_deconv(input_image_shape, use_color_augmentation=False, use_geometrical_augmentation=False):
    image_input = tf.keras.Input(shape=(input_image_shape))
    # input_noise = tf.keras.Input(shape=(2,))

    x = image_input

    #if use_color_augmentation:
    #    x = ColorAugmentation()(x)
    #if use_geometrical_augmentation:
    #    x = GeometricalAugmentation()(x, input_noise)

    encoded_img = model_encoder(input_image_shape)(x)
    decoded_img = model_decoder((15,15,128))(encoded_img)

    #if use_geometrical_augmentation:
    #    model = AutoEncoderModel(inputs=[image_input, input_noise], outputs=[decoded_img], name='autoencoder')
    #else:
    #    model = Model(inputs=[image_input], outputs=[decoded_img], name='autoencoder')

    model = Model(inputs=[image_input], outputs=[decoded_img], name='autoencoder')
    model.summary()

    return model

class Trainer:
    def __init__(self, model,
                     train_dataset,
                     val_dataset,
                     #load_weight=False,
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
                                                             save_best_only=False)

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
                                                      histogram_freq = 1,
                                                      profile_batch = '15,25')
        return cp_callback, early_stop, reduce_lr, profiler

    def train(self, epochs=300, early_stop_patience=100, reduce_lr_patience=50):
        xs = self.train_ds[0]
        ys = self.train_ds[1]
        val_xs = self.val_ds[0]
        val_ys = self.val_ds[1]

        start = time.time()
        callbacks = self.prepare_callbacks(early_stop_patience, reduce_lr_patience)

        history = self.model.fit(xs, ys,
                                     batch_size=self.batch_size,
                                     epochs=epochs,
                                     callbacks=callbacks,
                                     validation_data=(val_xs,val_ys),
                                     shuffle=True)

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))

def visualize_forcemaps(force_distribution, zaxis_first=False):
    f = force_distribution / np.max(force_distribution)
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.1)

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
            ax.imshow(f[:,:,p], cmap='gray', vmin=0, vmax=1.0)

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


# train_data, valid_data, test_data = load_data()
# model = model_conv_deconv(valid_data[0][0].shape)

# for training
# trainer = Trainer(model, train_data, valid_data)

# for test
# tester = Tester(model, test_data, 'ae_cp.basket-filling.autoencoder.20221018175305')
# tester.predict()
