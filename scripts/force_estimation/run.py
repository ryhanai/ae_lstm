# -*- coding: utf-8 -*-

import os
import time
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import argparse
from core.utils import *

import forcemap
from force_estimation_data_loader import ForceEstimationDataLoader
import force_estimation_v2_1 as fe


dataset = 'basket-filling2'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62

fmap = forcemap.GridForceMap('seria_basket')

dl = ForceEstimationDataLoader(os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset),
                               os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset+'-real'),
                               image_height=image_height,
                               image_width=image_width,
                               start_seq=1,
                               n_seqs=1800,  # n_seqs=1500,
                               start_frame=3, n_frames=3,
                               real_start_frame=1, real_n_frames=294
                               )


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
        self.opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        if train_dataset:
            self.train_ds = train_dataset

        if val_dataset:
            self.val_ds = val_dataset
            self.val_data_loaded = True

        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'runs')
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

        # profiler = tf.keras.callbacks.TensorBoard(log_dir='logs',
        #                                           histogram_freq=1,
        #                                           profile_batch='15,25')
        return cp_callback, early_stop, reduce_lr  # , profiler

    def train(self, epochs=300, early_stop_patience=100, reduce_lr_patience=50):
        xs = self.train_ds[0].astype('float32')
        ys = self.train_ds[1].astype('float32')
        val_xs = self.val_ds[0].astype('float32')
        val_ys = self.val_ds[1].astype('float32')

        start = time.time()
        callbacks = self.prepare_callbacks(early_stop_patience, reduce_lr_patience)

        source_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(xs)).batch(self.batch_size)
        # target_dataset = tf.data.Dataset.from_tensor_slices(xs_real).shuffle(len(x_real)).repeat().batch(self.batch_size)
        # train_dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

        val_source_dataset = tf.data.Dataset.from_tensor_slices((val_xs, val_ys)).batch(self.batch_size)
        # # val_target_dataset = tf.data.Dataset.from_tensor_slices(val_xs).shuffle(BUFFER_SIZE).repeat().batch(self.batch_size)
        # val_dataset = tf.data.Dataset.zip((val_source_dataset, val_target_dataset))

        history = self.model.fit(source_dataset,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=val_source_dataset,
                                 shuffle=True)

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))


def visualize_result(f_prediction, f_label, rgb, filename=None):
    forcemap.plot_force_map(f_prediction)
    plt.savefig('prediction.png')
    p = plt.imread('prediction.png')[:, :, :3]
    p = p[60:300, 190:1450, :]
    if f_label is None:
        p = cv2.resize(p, (1890, 360))
        res = np.concatenate([rgb, p], axis=1)
    if f_label is not None:
        forcemap.plot_force_map(f_label, 'ground truth')
        plt.savefig('ground_truth.png')
        g = plt.imread('ground_truth.png')[:, :, :3]
        g = g[60:300, 190:1450, :]
        pg = np.concatenate([p, g], axis=0)
        rgb2 = np.ones((480, 512, 3))
        rgb2[60:420] = rgb  # rgb.shape == (360, 512, 3)
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
        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'runs')
        self.checkpoint_path = os.path.join(d, checkpoint_file, 'cp.ckpt')
        print('load weights from ', self.checkpoint_path)
        self.model.load_weights(self.checkpoint_path)
        self.test_data = test_data

    def predict(self, n):
        xs = self.test_data[0][n:n+1]
        # ys = self.test_data[1][n:n+1]
        y_pred = self.model.predict(xs)
        plt.imshow(xs[0])
        forcemap.plot_force_map(y_pred[0], 'prediction')
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
        if type(self.test_data) is tuple:
            xs = self.test_data[0][n:n+1]
            force_label = self.test_data[1][n]
        else:
            xs = self.test_data[n:n+1]
            force_label = None
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        visualize_result(y_pred_forces[0], force_label, xs[0], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, xs[0]

    def predict_force_from_rgb_with_img(self, rgb):
        xs = np.expand_dims(rgb, axis=0)
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        forcemap.plot_force_map(y_pred_forces[0], 'prediction')
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



parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--task', type=str, default='test-real')
parser.add_argument('-w', '--weight', type=str, default='ae_cp.basket-filling2.model_resnet.20221202165608')
args = parser.parse_args()
message('task = {}'.format(args.task))
message('weight = {}'.format(args.weight))


if args.task == 'train':
    model = fe.model_rgb_to_fmap_res50()
    train_data, valid_data = dl.load_data_for_rgb2fmap(train_mode=True)
    trainer = Trainer(model, train_data, valid_data)
elif args.task == 'adaptation':
    model = fe.model_rgb_to_fmap_res50()
    train_data, valid_data = dl.load_data_for_rgb2fmap_with_real(train_mode=True)
    trainer = Trainer(model, train_data, valid_data)
elif args.task == 'test':
    model = fe.model_rgb_to_fmap_res50()
    test_data = dl.load_data_for_rgb2fmap(test_mode=True)
    tester = Tester(model, test_data, args.weight)
elif args.task == 'test-real':
    model = fe.model_rgb_to_fmap_res50()
    test_data = dl.load_real_data_for_rgb2fmap(test_mode=True)
    tester = Tester(model, test_data, args.weight)
