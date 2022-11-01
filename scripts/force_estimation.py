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
import res_unet
from core.utils import *
import pprint

dataset = 'basket-filling'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]

def load1(seqNo, frameNo, resize=True):
    dataset_path = os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset)
    rgb_path = os.path.join(dataset_path, str(seqNo), 'rgb{:05d}.jpg'.format(frameNo))
    depth_path = os.path.join(dataset_path, str(seqNo), 'depth_zip{:05d}.pkl'.format(frameNo))
    seg_path = os.path.join(dataset_path, str(seqNo), 'seg{:05d}.png'.format(frameNo))            
    force_path = os.path.join(dataset_path, str(seqNo), 'force_zip{:05d}.pkl'.format(frameNo))
    bin_state_path = os.path.join(dataset_path, str(seqNo), 'bin_state{:05d}.pkl'.format(frameNo))
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = pd.read_pickle(depth_path, compression='zip')
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    force = pd.read_pickle(force_path, compression='zip')
    bin_state = pd.read_pickle(bin_state_path)
    crop = 128
    rgb = rgb[:,crop:-crop] # crop the right & left side
    depth = depth[:,crop:-crop]
    seg = seg[:,crop:-crop]
    force = force[:,:,:20]

    # resize images
    rgb = cv2.resize(rgb, (image_width, image_height))
    depth = cv2.resize(depth, (image_width, image_height))
    depth = np.expand_dims(depth, axis=-1)
    seg = seg[::2,::2]
    return rgb, depth, seg, force, bin_state

def show1(scene_data):
    rgb, depth, seg, force, bin_state = scene_data
    pprint.pprint(bin_state)
    plt.figure()
    plt.imshow(rgb)
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.figure()
    plt.imshow(seg)
    visualize_forcemaps(force)
    plt.show()

import res_unet

# def model_res_unet():
#     num_classes = 62
#     m = res_unet.res_unet(image_shape, [64,128,256], 3, 3, 2)
#     return m

class ForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(ForceEstimationModel, self).__init__(*args, **kargs)
        tracker_names = ['loss', 'dloss', 'sloss', 'floss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        xs, y_labels = data

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True) # Forward pass
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss[0])
        self.train_trackers['dloss'].update_state(loss[1])
        self.train_trackers['sloss'].update_state(loss[2])
        self.train_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        xs, y_labels = data

        y_pred = self(xs, training=False) # Forward pass
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss[0])
        self.test_trackers['dloss'].update_state(loss[1])
        self.test_trackers['sloss'].update_state(loss[2])
        self.test_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y_labels, y_pred):
        y_pred_depth, y_pred_logits, y_pred_force = y_pred
        depth_labels, seg_labels, force_labels = y_labels
        dloss = tf.reduce_mean(tf.square(depth_labels - y_pred_depth))
        seg_labels = tf.cast(seg_labels, dtype=tf.int32)
        sloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seg_labels, logits=y_pred_logits)
        sloss = tf.reduce_mean(sloss)
        floss = tf.reduce_mean(tf.square(force_labels - y_pred_force))
        loss = dloss + sloss + floss
        return loss,dloss,sloss,floss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())

def model_for_force_estimation(input_shape=input_image_shape, 
                                num_filters=[16,32,64],
                                kernel_size=3,
                                num_channels=3,
                                num_classes=62
                                ):
    input = tf.keras.Input(shape=input_shape + [num_channels])

    encoder_output = res_unet.encoder(input, num_filters, kernel_size)

    # bridge layer, number of filters is double that of the last encoder layer
    bridge = res_unet.res_block(encoder_output[-1], [num_filters[-1]*2], kernel_size, strides=[2,1], name='bridge')

    print(encoder_output[-1].shape)
    #decoder_output = res_unet.decoder(bridge, encoder_output, num_filters, kernel_size)
    out1,out2,out3 = res_unet.multi_head_decoder(bridge, encoder_output, num_filters, kernel_size, num_heads=3)

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
                                    outputs=[output_depth,output_seg,output_force],
                                    name='force_estimator')

    model.summary()
    return model

def load_data():
    def split(X):
        return np.split(X, [int(len(X)*.75), int(len(X)*.875)])
    
    start_seq = 1001
    n_seqs = 50
    n_frames = 6
    x, y = np.mgrid[start_seq:start_seq+n_seqs, 0:n_frames]
    data_ids = list(zip(x.ravel(), y.ravel()))
    # data_ids = np.random.permutation(data_ids)
    X_rgb = np.empty((n_seqs*n_frames, image_height, image_width, 3))
    Y_depth = np.empty((n_seqs*n_frames, image_height, image_width, 1))
    Y_seg = np.empty((n_seqs*n_frames, image_height, image_width))
    Y_force = np.empty((n_seqs*n_frames, 40, 40, 20))

    # train_states = []

    total_frames = n_seqs*n_frames
    for i in range(0, total_frames):
        print('\rloading ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
        seqNo, frameNo = data_ids[i]
        rgb, depth, seg, force, bs = load1(seqNo, frameNo)
        X_rgb[i] = rgb
        Y_depth[i] = depth
        Y_seg[i] = seg
        Y_force[i] = force
        # train_states.append(bs)

    X_rgb /= 255.
    X_rgb_train, X_rgb_valid, X_rgb_test = split(X_rgb)
    dmax = np.max(Y_depth)
    dmin = np.min(Y_depth)
    Y_depth = (Y_depth - dmin) / (dmax - dmin)
    Y_depth_train, Y_depth_valid, Y_depth_test = split(Y_depth)
    Y_seg_train, Y_seg_valid, Y_seg_test = split(Y_seg)
    # Y = np.log(Y+1)
    fmax = np.max(Y_force)
    Y_force /= fmax
    Y_force_train, Y_force_valid, Y_force_test = split(Y_force)

    return (X_rgb_train, (Y_depth_train, Y_seg_train, Y_force_train)), (X_rgb_valid, (Y_depth_valid, Y_seg_valid, Y_force_valid)), (X_rgb_test, (Y_depth_test, Y_seg_test, Y_force_test))

def model_simple_encoder(input_shape, noise_stddev=0.2, name='encoder'):
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

def model_simple_decoder(input_shape, name='decoder'):
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

    encoded_img = model_simple_encoder(input_image_shape)(x)
    decoded_img = model_simple_decoder((15,15,128))(encoded_img)

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
        return xs[0],ys[0],y_pred[0]


train_data, valid_data, test_data = load_data()
# model = model_conv_deconv(valid_data[0][0].shape)
model = model_for_force_estimation()

# for training
trainer = Trainer(model, train_data, valid_data)

# for test
# tester = Tester(model, test_data, 'ae_cp.basket-filling.autoencoder.20221018175305')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.force_estimator.20221028172410')
# tester.predict()
