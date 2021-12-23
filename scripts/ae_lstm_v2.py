# -*- coding: utf-8 -*-

import os, sys, glob, re, time, copy
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from utils import *
from generator import *

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

input_image_shape = (80, 160, 3)
input_image_size = input_image_shape[:2]
dense_dim = 512
latent_dim = 32
time_window_size = 20
dof = 7
batch_size = 32


def model_encoder():
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(0.2))(image_input)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    encoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_dim, activation='selu'))(x)

    encoder = keras.Model([image_input], encoder_output, name='encoder')
    encoder.summary()
    return encoder

def model_lstm():
    # stacked LSTM
    imgvec_input = tf.keras.Input(shape=(time_window_size, latent_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = latent_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(50)(x)
    x = tf.keras.layers.Dense(state_dim)(x)
    lstm_image_output = tf.keras.layers.Lambda(lambda x:x[:,:latent_dim], output_shape=(latent_dim,))(x)
    lstm_joint_output = tf.keras.layers.Lambda(lambda x:x[:,latent_dim:], output_shape=(dof,))(x)

    lstm = keras.Model([imgvec_input, joint_input], [lstm_image_output, lstm_joint_output], name='lstm')
    lstm.summary()
    return lstm

def model_decoder():
    channels = 64
    ivec_input = tf.keras.Input(shape=(latent_dim))
    x = tf.keras.layers.Dense(5*10*channels, activation='selu')(ivec_input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Reshape(target_shape=(5, 10, channels))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    decoder_output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    decoder = keras.Model(ivec_input, decoder_output, name='decoder')
    decoder.summary()
    return decoder

def model_ae_lstm():
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))

    encoded_img = model_encoder()(image_input)
    predicted_ivec, predicted_jvec = model_lstm()([encoded_img, joint_input])

    #encoded_roi = model_roi_encoder()([image_input, roi])
    #predicted_ivec, predicted_jvec = model_roi_lstm()([encoded_roi, joint_input])
    decoded_img = model_decoder()(predicted_ivec)
    model = tf.keras.Model(inputs=[image_input, joint_input],
                               outputs=[decoded_img, predicted_jvec],
                               name='dplmodel')
    model.summary()
    return model


class AE_LSTM_Trainer:

    def __init__(self):
        self.batch_size = batch_size
        self.time_window = time_window_size
        self.model = model_ae_lstm()

        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        # create checkpoint and save best weight
        self.checkpoint_path = "/home/ryo/Program/moonshot/ae_lstm/runs/ae_cp/cp.ckpt"

        self.model_loaded = False
        self.val_data_loaded = False

    def load_train_data(self):
        self.train_ds = Dataset()
        self.train_ds.load('reaching', groups=range(1,300), image_size=input_image_size)
        self.train_ds.preprocess(self.time_window)
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(self.train_ds.get(),
                                                  None,
                                                  batch_size=self.batch_size,
                                                  time_window_size=self.time_window,
                                                  add_roi=False)

    def load_val_data(self):
        self.val_ds = Dataset()
        self.val_ds.load('reaching', groups=range(300,350), image_size=input_image_size)
        self.val_ds.preprocess(self.time_window)
        val_generator = DPLGenerator()
        self.val_gen = val_generator.flow(self.val_ds.get(),
                                              None,
                                              batch_size=self.batch_size,
                                              time_window_size=self.time_window,
                                              add_roi=False)

    def train(self, epochs=100):
        self.load_train_data()
        self.load_val_data()
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
        under construction
        '''
        self.train(epochs)
        samples = self.sample_with_current_policy(self)
        for n in range(20):
            self.train_gen.set_generated_samples(samples)
            self.train(epochs)


    def test(self):
        self.prepare_for_test()
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions = self.model.predict(x)
        visualize_ds(y[0])
        visualize_ds(x[0][:,0,:,:,:])
        visualize_ds(predicted_images)
        plt.show()
        return x[1],y[1],predicted_joint_positions

    def test_joint(self):
        self.prepare_for_test()
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions = self.model.predict(x)
        data = np.concatenate((x[1], predicted_joint_positions[:,np.newaxis,:]), axis=1)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1)
        for joint_id in range(dof):
            ax = fig.add_subplot(8//2, 2, joint_id+1)
            ax.plot(np.transpose(data[:,:,joint_id]))
        #ax.axis('off')
        plt.show()

    def predict_sequence_closed(self, group=0, create_anim_gif=True):
        '''
        Closed execution using a trained model
        '''
        self.prepare_for_test()
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

    def predict_sequence_open(self, group_num=0):
        '''
        Open execution using a trained model and teaching data
        Returns three trajectories: input sequence, predicted sequence and label sequence
        '''
        res = []
        d = self.val_ds.data
        seq_len = len(d[group_num][1])
        ishape = d[group_num][1][0].shape
        jv_dim = d[group_num][0].shape[1]
        batch_size = 1
        batch_x_imgs = np.empty((batch_size, self.time_window, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size, ishape[0], ishape[1], ishape[2]))
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

    def prepare_for_test(self, load_val_data=True):
        if not self.val_data_loaded and load_val_data:
            self.load_val_data()
            self.val_data_loaded = True
        if not self.model_loaded:
            self.model.compile(loss='mse', optimizer=self.opt)
            self.model.load_weights(self.checkpoint_path)
            self.model_loaded = True

