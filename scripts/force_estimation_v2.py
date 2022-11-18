# -*- coding: utf-8 -*-

#from typing import Concatenate
import tensorflow as tf
from tensorflow import keras
import force_distribution_viewer
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
from tensorflow.keras.applications.resnet50 import ResNet50

dataset = 'basket-filling'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62

def load1(seqNo, frameNo, data_type, resize=True):
    dataset_path = os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset)
    crop = 128

    def load_rgb():
        rgb_path = os.path.join(dataset_path, str(seqNo), 'rgb{:05d}.jpg'.format(frameNo))
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgb = rgb[:,crop:-crop] # crop the right & left side
        rgb = cv2.resize(rgb, (image_width, image_height))
        return rgb
    
    def load_depth():
        depth_path = os.path.join(dataset_path, str(seqNo), 'depth_zip{:05d}.pkl'.format(frameNo))
        depth = pd.read_pickle(depth_path, compression='zip')
        depth = depth[:,crop:-crop]
        depth = cv2.resize(depth, (image_width, image_height))
        depth = np.expand_dims(depth, axis=-1)
        return depth

    def load_segmentation_mask():
        seg_path = os.path.join(dataset_path, str(seqNo), 'seg{:05d}.png'.format(frameNo))            
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)    
        seg = seg[:,crop:-crop]
        seg = seg[::2,::2] # resize
        return seg

    def load_force():
        force_path = os.path.join(dataset_path, str(seqNo), 'force_zip{:05d}.pkl'.format(frameNo))
        force = pd.read_pickle(force_path, compression='zip')
        force = force[:,:,:20]
        return force

    def load_bin_state():
        bin_state_path = os.path.join(dataset_path, str(seqNo), 'bin_state{:05d}.pkl'.format(frameNo))
        bin_state = pd.read_pickle(bin_state_path)
        return bin_state
    
    if data_type == 'rgb':
        return load_rgb()
    elif data_type == 'depth':
        return load_depth()
    elif data_type == 'segmentation-mask':
        return load_segmentation_mask()
    elif data_type == 'force':
        return load_force()
    elif data_type == 'bin-state':
        return load_bin_state()
    else:
        raise 'Unknow data type requested'

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

def augment(xs, ys):
    img = xs
    fmap = ys

    # color transformation on xs
    # brightness_max_delta=0.2
    # contrast_lower=0.8
    # contrast_upper=1.2
    hue_max_delta=0.05
    # img = tf.image.random_brightness(img, max_delta=brightness_max_delta)
    # img = tf.image.random_contrast(img, lower=contrast_lower, upper=contrast_upper)
    img = tf.image.random_hue(img, max_delta=hue_max_delta)

    # apply save transform to xs and ys
    batch_sz = tf.shape(xs)[0]
    height = tf.shape(xs)[1]
    width = tf.shape(xs)[2]
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

    return img,fmap

class ForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(ForceEstimationModel, self).__init__(*args, **kargs)
        # tracker_names = ['loss', 'dloss', 'sloss', 'floss']
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
            y_pred = self(xs, training=True) # Forward pass
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        # self.train_trackers['dloss'].update_state(loss[1])
        # self.train_trackers['sloss'].update_state(loss[2])
        # self.train_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        xs, y_labels = data

        y_pred = self(xs, training=False) # Forward pass
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        # self.test_trackers['dloss'].update_state(loss[1])
        # self.test_trackers['sloss'].update_state(loss[2])
        # self.test_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    # def compute_loss(self, y_labels, y_pred):
    #     '''
    #         compute_loss for rgb to fmap
    #     '''
    #     y_pred_depth, y_pred_logits, y_pred_force = y_pred
    #     depth_labels, seg_labels, force_labels = y_labels
    #     dloss = tf.reduce_mean(tf.square(depth_labels - y_pred_depth))
    #     seg_labels = tf.cast(seg_labels, dtype=tf.int32)
    #     sloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seg_labels, logits=y_pred_logits)
    #     sloss = tf.reduce_mean(sloss)
    #     floss = tf.reduce_mean(tf.square(force_labels - y_pred_force))
    #     loss = dloss + sloss + floss
    #     return loss,dloss,sloss,floss

    def compute_loss(self, y_labels, y_pred):
        '''
            compute_loss for rgb to fmap
        '''
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())

def model_rgb_to_fmap(input_shape=input_image_shape, 
                                num_filters=[16,32,64],
                                kernel_size=3,
                                num_channels=3,
                                num_classes=62,
                                noise_stddev=0.3,
                                ):
    input = tf.keras.Input(shape=input_shape + [num_channels])

    # Data Augmentation
    x = tf.keras.layers.GaussianNoise(noise_stddev)(input)

    encoder_output = res_unet.encoder(x, num_filters, kernel_size)

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


def split(X):
    return np.split(X, [int(len(X)*.75), int(len(X)*.875)])

def load_data(rgb=True, depth=True, segmentation_mask=True, force=True, bin_state=False):
    start_seq = 1
    #n_seqs = 20
    #n_seqs = 450
    n_seqs = 1500
    start_frame = 3
    n_frames = 3
    x, y = np.mgrid[start_seq:start_seq+n_seqs, start_frame:start_frame+n_frames]
    data_ids = list(zip(x.ravel(), y.ravel()))
    total_frames = n_seqs*n_frames

    if rgb:
        X_rgb = np.empty((n_seqs*n_frames, image_height, image_width, 3))
        for i in range(0, total_frames):
            print('\rloading RGB ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            seqNo, frameNo = data_ids[i]
            X_rgb[i] = load1(seqNo, frameNo, data_type='rgb')
        X_rgb /= 255.
    else:
        X_rgb = []

    if depth:
        Y_depth = np.empty((n_seqs*n_frames, image_height, image_width, 1))
        for i in range(0, total_frames):
            print('\rloading depth ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            seqNo, frameNo = data_ids[i]
            Y_depth[i] = load1(seqNo, frameNo, data_type='depth')
        dmax = np.max(Y_depth)
        dmin = np.min(Y_depth)
        Y_depth = (Y_depth - dmin) / (dmax - dmin)
    else:
        Y_depth = []

    if segmentation_mask:
        Y_seg = np.empty((n_seqs*n_frames, image_height, image_width))
        for i in range(0, total_frames):
            print('\rloading segmentation mask ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            seqNo, frameNo = data_ids[i]
            Y_seg[i] = load1(seqNo, frameNo, data_type='segmentation-mask')
    else:
        Y_seg = []

    if force:
        Y_force = np.empty((n_seqs*n_frames, 40, 40, 20))
        for i in range(0, total_frames):
            print('\rloading force ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            seqNo, frameNo = data_ids[i]
            Y_force[i] = load1(seqNo, frameNo, data_type='force')
        fmax = np.max(Y_force)
        Y_force /= fmax
    else:
        Y_force = []

    bin_states = []
    if bin_state:
        for i in range(0, total_frames):
            print('\rloading bin_state ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            seqNo, frameNo = data_ids[i]
            bin_states.append(load1(seqNo, frameNo, data_type='bin-state'))
    
    return X_rgb, Y_depth, Y_seg, Y_force, bin_states


def model_simple_decoder(input_shape, name='decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))

    x = feature_input
    x = deconv_block(x, 1024, with_upsampling=False) # 12x16
    x = deconv_block(x, 512) # 24x32
    x = deconv_block(x, 256, with_upsampling=False) # 24x32
    x = deconv_block(x, 128, with_upsampling=False) # 24x32
    x = deconv_block(x, 64) # 48x64
    x = deconv_block(x, 32, with_upsampling=False) # 48x64
    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder

def model_resnet_decoder(input_shape, name='resnet_decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [1024,512], 3, strides=[1,1], name='resb1')
    x = res_unet.res_block(x, [256,128], 3, strides=[1,1], name='resb2')
    x = res_unet.upsample(x, (24,32))
    x = res_unet.res_block(x, [64,64], 3, strides=[1,1], name='resb3')
    x = res_unet.res_block(x, [32,32], 3, strides=[1,1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder

def model_resnet_decoder2(input_shape, name='resnet_decoder2'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [256,128], 3, strides=[1,1], name='resb1')
    x = res_unet.res_block(x, [64,64], 3, strides=[1,1], name='resb2')
    x = res_unet.upsample(x, (24,32)) # (46,64)
    x = res_unet.res_block(x, [32,32], 3, strides=[1,1], name='resb4')

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
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0,1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    resnet50 = ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)
    decoded_img = model_resnet_decoder((12,16,2048))(encoded_img)

    # model = Model(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    model = ForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    model.summary()

    return model

def model_depth_to_fmap(input_shape=input_image_shape, kernel_size=3, num_classes=num_classes):
    input_depth = tf.keras.Input(shape=input_shape + [1])
    input_seg = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=num_classes))(input_seg)
    x = tf.keras.layers.Concatenate(axis=-1)([input_depth, x])

    encoder_output = res_unet.encoder(x, num_filters=(32,64,64,128,256), kernel_size=kernel_size)
    decoded_img = model_resnet_decoder2((23,32,256))(encoder_output[-1])

    model = Model(inputs=[input_depth,input_seg], outputs=[decoded_img], name='model_depth_to_fmap')
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

def visualize_forcemaps(force_distribution, title='', zaxis_first=False):
    f = force_distribution / np.max(force_distribution)
    fig = plt.figure(figsize=(16,4))
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
            ax.imshow(f[:,:,p], cmap='gray', vmin=0, vmax=1.0)

def visualize_result(f_prediction, f_label, rgb, filename=None):
    visualize_forcemaps(f_prediction, title='prediction')
    plt.savefig('prediction.png')
    visualize_forcemaps(f_label, title='ground truth')
    plt.savefig('ground_truth.png')
    p = plt.imread('prediction.png')[:,:,:3]
    g = plt.imread('ground_truth.png')[:,:,:3]
    pg = np.concatenate([p,g], axis=0)
    rgb2 = np.ones((800,512,3))
    #rgb = cv2.resize(rgb, (256,180))
    rgb2[220:580] = rgb
    res = np.concatenate([rgb2,pg], axis=1)
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(res)
    if filename != None:
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
        return xs[0],ys[0],y_pred[0]

    def predict_force_from_rgb(self, n):
        xs = self.test_data[0][n:n+1]
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        force_label = self.test_data[1][n]

        visualize_result(y_pred_forces[0], force_label, xs[0], 'result{:05d}.png'.format(n))
        #plt.figure()
        #plt.imshow(xs[0])
        #visualize_forcemaps(force_label)
        #visualize_forcemaps(y_pred_forces[0])
        #plt.show()
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
        xs = xs_depth,xs_seg
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        force_label = self.test_data[1][n]

        visualize_result(y_pred_forces[0], force_label, rgb[n], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, rgb[n]


def load_data_for_rgb2fmap():
    rgb, depth, seg, force, _  = load_data(rgb=True, depth=False, segmentation_mask=False, force=True)
    train_rgb, valid_rgb, test_rgb = split(rgb)
    del rgb
    train_force, valid_force, test_force = split(force)
    del force
    #train_bin_state, valid_bin_state, test_bin_state = split(bin_state)
    #del bin_state
    train_data = train_rgb, train_force
    valid_data = valid_rgb, valid_force
    test_data = test_rgb, test_force
    return train_data,valid_data,test_data

viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

def show_bin_state(fcam, bin_state, fmap, draw_fmap=True, draw_force_gradient=False):
    fv = np.zeros((40,40,40))
    fv[:,:,:20] = fmap
    positions = fcam.positions
    viewer.publish_bin_state(bin_state, positions, fv, draw_fmap=draw_fmap, draw_force_gradient=draw_force_gradient)

def load_data_for_dseg2fmap():
    rgb, depth, seg, force = load_data(rgb=True, depth=True, segmentation_mask=True, force=True)
    train_depth, valid_depth, test_depth = split(depth)
    del depth
    train_seg, valid_seg, test_seg = split(seg)
    del seg
    train_force, valid_force, test_force = split(force)
    del force
    train_data = (train_depth, train_seg), train_force
    valid_data = (valid_depth, valid_seg), valid_force
    test_data = (test_depth, test_seg), test_force
    return train_data,valid_data,test_data

def load_bin_states():
    _,_,_,_,bin_states = load_data(rgb=False, depth=False, segmentation_mask=False, force=False, bin_state=True)
    return split(bin_states)

import scipy.linalg

def pick_direction_plan(fcam, fmap, bin_state, gp=[0.02, -0.04, 0.79], radius=0.05, scale=[0.005,0.01,0.004]):
    gp = np.array(gp)
    fv = np.zeros((40,40,40))
    fv[:,:,:20] = fmap
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fcam.positions
    idx = np.where(np.sum((ps - gp)*g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]
    pos_val_pairs = [(p,g) for (p,g) in zip(fps, fg_vecs) if scipy.linalg.norm(g) > 0.008]
    pz,vz = zip(*pos_val_pairs)
    pz = np.array(pz)
    vz = np.array(vz)
    viewer.publish_bin_state(bin_state, ps, fv, draw_fmap=False, draw_force_gradient=False)
    viewer.draw_vector_field(pz, vz, scale=0.3)
    viewer.rviz_client.draw_sphere(gp,[1,0,0,1],[0.01,0.01,0.01])
    viewer.rviz_client.show()
    idx = np.where(scipy.linalg.norm(pz - gp, axis=1) < radius)[0]
    pick_direction = np.sum(vz[idx], axis=0)
    pick_direction /= np.linalg.norm(pick_direction)
    viewer.rviz_client.draw_arrow(gp, gp + pick_direction * 0.1, [0,1,0,1], scale)
    viewer.rviz_client.show()
    return pz,vz,pick_direction


# model = model_conv_deconv()
model = model_rgb_to_fmap_res50()
# model = model_depth_to_fmap()

train_data,valid_data,test_data = load_data_for_rgb2fmap()
train_bs,valid_bs,test_bs = load_bin_states()


# for real-world test
# test_data = None

def load_real(n):
    file_path = os.path.join(os.getenv('HOME'), 'Dataset/dataset2/basket-filling-real', 'frame{:05d}.png'.format(n))
    img = plt.imread(file_path)
    c = (-40,25)
    crop = 64
    img2 = img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]
    return img2

# for training
# trainer = Trainer(model, train_data, valid_data)

# for test
# tester = Tester(model, test_data, 'ae_cp.basket-filling.force_estimator.best')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.force_estimator.20221028172410')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221101213121')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_depth_to_fmap.20221104234049')

# best
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_depth_to_fmap.20221107110203')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221107144923')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221115154455')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221115182656')
tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221115193122')

# gaussian / color augmentation
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221108181626')
