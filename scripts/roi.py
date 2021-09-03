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

# from keras.layers import Conv2D, MaxPooling2D, Dense, Input, UpSampling2D, BatchNormalization
# from keras.layers import merge, concatenate
# from keras.callbacks import EarlyStopping, TensorBoard
# from keras.layers import GaussianNoise

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
    

def preprocess_image(img):
    # crop the center region
    img = img[130:250, 120:360] 
    img = cv2.resize(img, (width, height))
    return img


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
                 image=True,
                 visualize=False,
                 start_step=0,
                 step_length=None):

    def load_group(group):
        path = os.path.join(dataset_path, '%s/group%d'%(action, group))

        # load joint and frame indices
        joint_file = os.path.join(path, 'torobo_joint_position_avg.txt')
        joint_seq_time = load_torobo_unity_joint_seq(joint_file, start_step, step_length)
        joint_seq = np.delete(joint_seq_time, [0,8], axis=1) # delete wall time & something

        # load images & preprocess them
        n_frames = joint_seq.shape[0]
        frames = []
        for i in range(n_frames):
            img = plt.imread(os.path.join(path, 'image_frame%d.jpg'%i))
            # do not preprocess in ROI learning
            # img = preprocess_image(img)
            frames.append(img)
        if visualize:
            visualize_ds(frames)

        return joint_seq, frames
        
    start = time.time()
    data = []
    for group in groups:
        data.append(load_group(group))
        

    end = time.time()
    print('total time spent {}'.format((end-start)/60))

    return data

# def load_joint_dataset(action='pushing', groups=range(1,2), start_step=0, step_length=None):
#     def load_group(group):
#         joint_file = os.path.join(dataset_path, '%s/group%d'%(action, group), 'torobo_joint_position_avg.txt')
#         joint_seq_time = load_torobo_unity_joint_seq(joint_file, start_step, step_length)
#         joint_seq = np.delete(joint_seq_time, [0,8], axis=1) # delete wall time & something
#         return joint_seq
    
#     joint_seqs = [load_group(g) for g in groups]
#     return joint_seqs

def gen_input_data(joint_seqs, time_window=20):
    n_inputs = np.sum([js.shape[0] - time_window for js in joint_seqs])
    dim_input = joint_seqs[0].shape[1]
    data = np.zeros((n_inputs, time_window, dim_input))
    labels = np.zeros((n_inputs, dim_input))
    print(data.shape)

    n = 0
    for joint_seq in joint_seqs:
        for i in range(joint_seq.shape[0] - time_window):
            data[n] = joint_seq[i:i+time_window]
            labels[n] = joint_seq[i+time_window]
            n += 1
    return data,labels

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


model = model_crop_and_resize()
imgs = np.array(load_dataset(groups=[2])[0][1]) / 255
rois = np.tile([0.2, 0.2, 0.8, 0.8], [imgs.shape[0], 1])
result = model([imgs, rois])


class JointLSTM(tf.keras.Model):

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

class ROI_AE(tf.keras.Model):

    def __init__(self, dof=8):
        super(ROI_AE, self).__init__()

        self._maxlen = 20
        self._dof = dof

        self.lstm = tf.keras.Sequential([
            tf.keras.layers.Cropping2D(),
            tf.keras.layers.Resizing(
                height, width, interpolation='bilinear', crop_to_aspect_ratio=False),
            #tf.keras.layers.InputLayer(batch_input_shape=(None, self._maxlen, self._dof)),
            tf.keras.layers.InputLayer(input_shape=(None, self._dof)),
            tf.keras.layers.LSTM(self._dof, return_sequences=True), # stacked LSTM
            tf.keras.layers.LSTM(self._dof, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(self._dof)
        ])

    def call(self, x):
        return self.lstm(x)



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
