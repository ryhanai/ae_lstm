# -*- coding: utf-8 -*-

import os, sys, glob, re, time
import cv2
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from generator import *

# from keras.layers import Conv2D, MaxPool2D, Dense, Input, UpSampling2D, BatchNormalization, Flatten
# from keras.layers import merge, concatenate
# from keras.callbacks import EarlyStopping, TensorBoard


def numerical_sort(value):
    """
    Splits out any digits in a filename, turns it into an actual number, and returns the result for sorting.
    :param value: filename
    :return:

    author: kei
    date: 20190903
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def visualize_ds(images, max_samples=20):
    samples = min(len(images), max_samples)

    fig = plt.figure(figsize=(10,samples))
    fig.subplots_adjust(hspace=0.1)
  
    for p in range(samples):
        ax = fig.add_subplot(samples//4, 4, p+1)
        ax.axis('off')
        ax.imshow(images[p])
    

dataset_path = '/home/ryo/Dataset/dataset'
sys.path.append(dataset_path)

height = 80
width = 160

def load_torobo_unity_joint_seq(joint_seq_file, start_step=0, step_length=None):
    joint_seq = []
    joint_time = []
    with open(joint_seq_file) as f:
        contents = f.readlines()
        if step_length == None:
            step_length = len(contents)
        for line_idx in range(start_step, start_step+step_length):
            line = contents[line_idx]
            line = line.rstrip('\n')
            line = line.rstrip(', )')
            line = line.split(':')
            time = line[2].rstrip('\tPOSITION')
            time_arr = np.fromstring(time, dtype=np.float64, sep=' ')
            data = line[3].rstrip(' )')
            data = data.lstrip(' (')
            data_arr = np.fromstring(data, dtype=np.float64, sep=', ')
            joint_time.append(time_arr)
            joint_seq.append(data_arr)
    np_joint_time = np.array(joint_time)
    np_joint_seq = np.array(joint_seq)
    np_joint_seq_time = np.concatenate([np_joint_time, np_joint_seq], 1)

    return np_joint_seq_time


def load_dataset(action='pushing',
                 groups=range(1,6),
                 load_image=True,
                 sampling_interval=1,
                 visualize=False,
                 start_step=0,
                 step_length=None,
                 normalize=True):
    
    def load_group(group, joint_range=None):
        path = os.path.join(dataset_path, '%s/group%d'%(action, group))

        # load joint and frame indices
        joint_file = os.path.join(path, 'torobo_joint_position_avg.txt')
        joint_seq_time = load_torobo_unity_joint_seq(joint_file, start_step, step_length)
        joint_seq = np.delete(joint_seq_time, [0,8], axis=1) # delete wall time & something

        if normalize:
            jmin, jmax = joint_range
            joint_seq = (joint_seq - jmin) / (jmax - jmin)
        
        # load images
        n_frames = joint_seq.shape[0]
        if load_image:
            frames = []
            for i in range(0, n_frames, sampling_interval):
                img = plt.imread(os.path.join(path, 'image_frame%d.jpg'%i))
                if normalize:
                    img = img/255.
                frames.append(img)
                
            return joint_seq, frames
        else:
            return joint_seq
        
    start = time.time()
    data = []

    if normalize:
        jp_range = joint_position_range()

    n_groups = len(groups)
    for i,group in enumerate(groups):
        print('\rloading: {}/{}'.format(i, n_groups), end='')
        if normalize:
            data.append(load_group(group, jp_range))
        else:
            data.append(load_group(group))

    end = time.time()
    print('\ntotal time spent for loading data: {} [min]'.format((end-start)/60))

    return data

def joint_position_range():
    ds = load_dataset(groups=range(1,400), load_image=False, normalize=False)
    joint_max_positions = np.max([np.max(group, axis=0) for group in ds], axis=0)
    joint_min_positions = np.min([np.min(group, axis=0) for group in ds], axis=0)
    return joint_min_positions, joint_max_positions

def generate_dataset_for_AE_training(ds):
    images = []
    for group in ds:
        joint_seq, frames = group
        images.extend(frames)
    input_images = np.array(images) / 255.0
    labels = np.zeros((input_images.shape[0], 80, 160, 3))
    roi = np.array([0.48, 0.25, 0.92, 0.75]) # [y1, x1, y2, x2] in normalized coodinates
    bboxes = np.tile(roi, [input_images.shape[0], 1])
    return (input_images, bboxes), labels

def hand_coded_ROI_policy():
    return "a sequence of ROIs"

def fixed_ROI_policy(ds):
    roi = np.array([0.48, 0.25, 0.92, 0.75])
    n_data = [group[0].shape[0] for group in ds]
    return np.tile(roi, [ds.shape[0], 1])
    

from scipy.linalg import norm

def print_distances(joint_vec_seq):
    for i in range(len(joint_vec_seq)-1):
        print(norm(joint_vec_seq[i] - joint_vec_seq[i+1]))

        
def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, (height, width))

def model_crop_and_resize():
    image_input = tf.keras.Input(shape=(270, 480, 3))
    roi_input = tf.keras.Input(shape=(4, ))
    cropped = tf.keras.layers.Lambda(crop_and_resize, output_shape=(height, width)) ([image_input, roi_input])
    return tf.keras.Model([image_input, roi_input], cropped, name='test')

def test_crop_and_resize():
    model = model_crop_and_resize()
    imgs = np.array(load_dataset(groups=[2])[0][1]) / 255
    rois = np.tile([0.2, 0.2, 0.8, 0.8], [imgs.shape[0], 1])
    result = model([imgs, rois])
    plt.imshow(result[50])
    plt.show()


loss_tracker = keras.metrics.Mean(name="loss")
val_loss_tracker = keras.metrics.Mean(name="val_loss")

class AutoencoderWithCrop(tf.keras.Model):
    """
    override loss computation
    """
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            loss = self.compute_loss(y, y_pred, x)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass
        val_loss = self.compute_loss(y, y_pred, x)

        val_loss_tracker.update_state(val_loss)
        return {"loss": val_loss_tracker.result()}

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint, x_roi = x
        y_image, y_joint = y
        y_image_cropped = crop_and_resize((y_image, x_roi[:,-1]))
        
        image_loss = tf.reduce_mean(tf.square(y_image_cropped - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        loss = image_loss + 10.*joint_loss
        #loss = keras.losses.mean_squared_error(y_image_cropped, y_pred[0])
        return loss
    
    @property
    def metrics(self):
        return [loss_tracker, val_loss_tracker]


dense_dim = 512
latent_dim = 32

def model_roi_ae_lstm(time_window_size=20):
    # encoder
    image_input = tf.keras.Input(shape=(time_window_size, 270, 480, 3))
    roi_input = tf.keras.Input(shape=(time_window_size, 4))    
    roi = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(crop_and_resize, output_shape=(height, width, 3))) ([image_input, roi_input])

    roi_extractor = tf.keras.Model([image_input, roi_input], roi, name='roi_extractor')
    roi_extractor.summary()
    
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(0.2))(roi)

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

    # stacked LSTM
    dof = 8
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = latent_dim + dof
    x = tf.keras.layers.concatenate([encoder_output, joint_input])
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(50)(x)
    x = tf.keras.layers.Dense(state_dim)(x)
    lstm_image_output = tf.keras.layers.Lambda(lambda x:x[:,:latent_dim], output_shape=(latent_dim,))(x)
    lstm_joint_output = tf.keras.layers.Lambda(lambda x:x[:,latent_dim:], output_shape=(dof,))(x)

    # decoder
    channels = 64
    x = tf.keras.layers.Dense(5*10*channels, activation='selu')(lstm_image_output)
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

    encoder = keras.Model([image_input, roi_input], encoder_output, name='encoder')
    encoder.summary()

    model = AutoencoderWithCrop([image_input, joint_input, roi_input],
                                [decoder_output, lstm_joint_output],
                                name='dplmodel')
    model.summary()
    
    # subtracted = tf.keras.layers.subtract([roi, decoder_output])
    # ae_sub = keras.Model([image_input, joint_input, roi_input], subtracted, name='ae_sub')
    # ae_sub.summary()
    
    return roi_extractor, encoder, model


def show_images(original_images, reconstructed_images, n=10):
    idx = [np.random.randint(1,20) for i in range(n)]
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.title("original")
        plt.imshow(original_images[idx[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        cx = plt.subplot(3, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(reconstructed_images[idx[i]])
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)

    plt.show()

    
class ROI_AE_LSTM_Trainer:

    def __init__(self):
        self.batch_size = 32
        self.time_window = 10
        self.roi_extractor, self.encoder, self.model = model_roi_ae_lstm(time_window_size=self.time_window)

        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        # create checkpoint and save best weight
        self.checkpoint_path = "/home/ryo/Program/ae_lstm/runs/ae_cp/cp.ckpt"

    def load_train_data(self):
        # pushing: group1 - group400
        train_ds = load_dataset(groups=range(1,300))
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(train_ds, batch_size=self.batch_size, time_window_size=self.time_window)

    def load_val_data(self):
        val_ds = load_dataset(groups=range(300,350))
        val_generator = DPLGenerator()
        self.val_gen = val_generator.flow(val_ds, batch_size=self.batch_size, time_window_size=self.time_window)

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

        # train the model
        total_train = self.train_gen.number_of_data()
        total_val = self.val_gen.number_of_data()
        history = self.model.fit(self.train_gen,
                                 steps_per_epoch=total_train // self.batch_size,
                                 epochs=epochs,
                                 validation_data=self.val_gen,
                                 validation_steps=total_val // self.batch_size,
                                 callbacks=[cp_callback, early_stop, reduce_lr])

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))

    def test(self):
        self.load_val_data()
        self.model.compile(loss='mse', optimizer=self.opt)
        self.model.load_weights(self.checkpoint_path)
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions = self.model.predict(x)
        visualize_ds(predicted_images)
        # roi_images = self.roi_extractor.predict((x[0],x[2]))
        # visualize_ds(roi_images[:,-1,:,:,:])
        roi_images = crop_and_resize((y[0], x[2][:,-1]))
        visualize_ds(roi_images)
        plt.show()

    def test_joint(self):
        self.load_val_data()
        self.model.compile(loss='mse', optimizer=self.opt)
        self.model.load_weights(self.checkpoint_path)
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions = self.model.predict(x)
        data = np.concatenate((x[1], predicted_joint_positions[:,np.newaxis,:]), axis=1)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1)
        for joint_id in range(8):
            ax = fig.add_subplot(8//2, 2, joint_id+1)
            ax.plot(np.transpose(data[:,:,joint_id]))
        #ax.axis('off')
        plt.show()
        
    def generate_sequence(self):
        # load best checkpoint and evaluate
        self.model.compile(loss='mse', optimizer=self.opt)
        self.model.load_weights(self.checkpoint_path)
        
        reconstructed_images = self.model.predict(self.val_data)

        show_images(roi_images, reconstructed_images)

        return reconstructed_images, roi_images


class LSTM_trainer:

    def __init__(self, dof=8):
        super(JointLSTM, self).__init__()

        self._maxlen = 20
        self._dof = dof

        self.lstm = tf.keras.Sequential([
            #tf.keras.layers.InputLayer(batch_input_shape=(None, self._maxlen, self._dof)),
            tf.keras.layers.InputLayer(input_shape=(None, self._dof)),
            tf.keras.layers.LSTM(self._dof, return_sequences=True), # stacked LSTM
            tf.keras.layers.LSTM(self._dof, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(self._dof)
        ])

    def call(self, x):
        return self.lstm(x)

    
def train_all():
    ae_trainer = AE_Trainer()
    lstm_trainer = LSTM_Trainer()
    ae_trainer.train()
    lstm_trainer.train()


# train_data,train_labels = gen_input_data(load_dataset(groups=range(1,10)))
# val_data,val_labels = gen_input_data(load_dataset(groups=range(10,15)))
# opt = keras.optimizers.Adamax(learning_rate=0.001)
# lstm = JointLSTM()
# lstm.compile(loss='mse', optimizer=opt)

# def train():
#     start = time.time()

#     # Create a callback that saves the model's weights
#     # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#     #                                                  save_weights_only=True,
#     #                                                  verbose=1,
#     #                                                  mode='min',
#     #                                                  save_best_only=True)

#     # early stopping if not changing for 50 epochs
#     # early_stop = EarlyStopping(monitor='val_loss',
#     #                            patience=100)

#     # reduce learning rate
#     # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#     #                                                  factor=0.1,
#     #                                                  patience=50, 
#     #                                                  verbose=1,
#     #                                                  min_lr=0.00001)

#     # train the model
#     history = lstm.fit(train_data,
#                        train_labels, 
#                        epochs=50,
#                        batch_size=32,
#                        shuffle=True,
#                        validation_data=(val_data, val_labels),
#                        callbacks=[])
#                        #callbacks=[cp_callback, early_stop, reduce_lr])

#     end = time.time()
#     print('\ntotal time spent {}'.format((end-start)/60))

#     lstm.lstm.summary()
    
#     plt.plot(history.epoch, history.history['loss'], label='train_loss')
#     plt.plot(history.epoch, history.history['val_loss'], label='test_loss')
#     plt.title('Epochs on Training Loss')
#     plt.xlabel('# of Epochs')
#     plt.ylabel('Mean Squared Error')
#     plt.legend()
#     plt.show()
#     return history
