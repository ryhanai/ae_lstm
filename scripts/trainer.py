# -*- coding: utf-8 -*-

import os, sys, glob, re, time, copy
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils import *
import generator

class Trainer:

    def __init__(self, model,
                     train_dataset,
                     val_dataset,
                     load_weight=False,
                     time_window_size=20,
                     batch_size=32,
                     runs_directory=None,
                     checkpoint_file=None):

        self.input_image_shape = val_dataset.data[0][1][0].shape
        self.time_window = time_window_size
        self.batch_size = batch_size

        self.model = model
        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        if train_dataset:
            self.train_ds = train_dataset
            self.train_gen = generator.DPLGenerator().flow(train_dataset.get(),
                                                               None,
                                                               batch_size=self.batch_size,
                                                               time_window_size=self.time_window,
                                                               add_roi=False)

        if val_dataset:
            self.val_ds = val_dataset
            self.val_gen = generator.DPLGenerator().flow(val_dataset.get(),
                                                             None,
                                                             batch_size=self.batch_size,
                                                             time_window_size=self.time_window,
                                                             add_roi=False)
            self.val_data_loaded = True

        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.getcwd()), 'runs')
        f = checkpoint_file if checkpoint_file else 'ae_cp.{}.{}.{}'.format(val_dataset.name, self.model.name, datetime.now().strftime('%Y%m%d%H%M%S'))
        self.checkpoint_path = os.path.join(d, f, 'cp.ckpt')

        if checkpoint_file:
            print('load weights from ', checkpoint_file)
            self.model.load_weights(self.checkpoint_path)

    def train(self, epochs=100):
        start = time.time()

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         mode='min',
                                                         save_best_only=True)

        # early stopping if not changing for 50 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=100)

        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=50,
                                                         verbose=1,
                                                         min_lr=0.00001)

        profiler = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                  histogram_freq = 1,
                                                  profile_batch = '15,25'
                                                  )

        # train the model
        total_train = self.train_gen.number_of_data()
        total_val = self.val_gen.number_of_data()
        history = self.model.fit(self.train_gen,
                                 steps_per_epoch=total_train // self.batch_size,
                                 epochs=epochs,
                                 validation_data=self.val_gen,
                                 validation_steps=total_val // self.batch_size,
                                 callbacks=[cp_callback, early_stop, reduce_lr, profiler])

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))

    def train_closed(self, epochs=20):
        '''
        under implementation
        '''
        self.train(epochs)

        samples = self.sample_with_current_policy(self)
        for n in range(20):
            self.train_gen.set_generated_samples(samples)
            self.train(epochs)


    def predict_images(self):
        x,y = next(self.val_gen)
        rois = []
        y_pred = self.model.predict(x)
        if len(y_pred) == 3:
            predicted_images, _, rois = y_pred
        else:
            predicted_images, _ = y_pred
        visualize_ds(y[0], rois)
        visualize_ds(x[0][:,0,:,:,:], rois)
        visualize_ds(predicted_images)
        plt.show()

    def predict_joint_angles(self):
        x,y = next(self.val_gen)
        y_pred = self.model.predict(x)
        predicted_joint_positions = y_pred[1]
        data = np.concatenate((x[1], predicted_joint_positions[:,np.newaxis,:]), axis=1)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1)
        dof = predicted_joint_positions.shape[1]
        for joint_id in range(dof):
            ax = fig.add_subplot(8//2, 2, joint_id+1)
            ax.plot(np.transpose(data[:,:,joint_id]))
        plt.show()

    def predict_sequence_closed(self, group=0, create_anim_gif=True):
        '''
        Closed(=Offline) execution using a trained model
        '''
        jseq, iseq = self.val_ds.data[group]
        batch_size = 1
        sm = StateManager(self.time_window, batch_size)
        sm.setHistory(np.array(iseq[:self.time_window]), jseq[:self.time_window])

        for j in range(len(jseq)-self.time_window):
            y = self.model.predict(sm.getHistory())
            predicted_image = y[0][0]
            predicted_joint_position = y[1][0]
            sm.addState(predicted_image, predicted_joint_position)

        if create_anim_gif:
            create_anim_gif_from_images(sm.getFrameImages(), 'closed_exec{:05d}.gif'.format(group))
        return sm

    def predict_for_group(self, group_num=0):
        '''
        Do prediction for each point in a given group using a trained model and teaching data.
        Returns three trajectories: input sequence, predicted sequence and label sequence.
        '''
        res = []
        d = self.val_ds.data
        seq_len = len(d[group_num][1])
        ishape = d[group_num][1][0].shape
        jv_dim = d[group_num][0].shape[1]
        batch_size = 1
        batch_x_imgs = np.empty((batch_size, self.time_window) + ishape)
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size,) + ishape)
        batch_y_jv = np.empty((batch_size, jv_dim))

        for seq_num in range(seq_len-self.time_window):
            print(seq_num)
            batch_x_jvs[:] = d[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[:] = d[group_num][0][seq_num+self.time_window]
            batch_x_imgs[:] = d[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[:] = d[group_num][1][seq_num+self.time_window]

            predicted_images, predicted_joint_positions = self.model.predict((batch_x_imgs, batch_x_jvs))
            current_joint_position = copy.copy(batch_x_jvs[0][-1])
            label_joint_position = copy.copy(batch_y_jv[0])
            predicted_joint_position = predicted_joint_positions[0]
            res.append((current_joint_position, predicted_joint_position, label_joint_position))
        return list(zip(*res))

