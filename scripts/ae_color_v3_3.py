# -*- coding: utf-8 -*-
"""ae_color_v3.3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11eBSz3FHYxRLxHblDhWT_UjLJtItlY7b
"""

# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# !rm -rf ./logs/

import os, sys, glob, re, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Dense, Input, UpSampling2D, BatchNormalization
#from keras.optimizers import Adam
#from keras.optimizer_v1 import Adamax
from keras.callbacks import EarlyStopping, TensorBoard


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
    

def process_ds(paths, vis=False):
    print('total images', paths)
    for p in paths:
        img = plt.imread(p)
        img = cv2.resize(img, (height, width))
        BATCH.append(img) #global variable

    if vis:
        visualize_ds(BATCH)


# clone sample images
# path = '/content/drive/MyDrive/ds/pushing'
path = '/home/ryo/Dataset/dataset'
sys.path.append(path)
#dirs = os.listdir(path)
dirs = ['pushing']

height = 160
width = 80
BATCH = []

def load_dataset(visualize=False):
    start = time.time()
    for dir in dirs:
        images = sorted(glob.glob(os.path.join(path, dir, 'group*', 'image_frame*.jpg'), recursive=True), key=numerical_sort)
        images = images[:3000]
        # images = sorted(glob.glob(os.path.join(path, dir, 'group45', 'image_frame*.jpg'), recursive=True), key=numerical_sort)
        process_ds(images, vis=visualize)

    end = time.time()
    print('total time spent {}'.format((end-start)/60))


load_dataset()
print(len(BATCH))

start = time.time()

ds = tf.stack(BATCH) #create tensor of samples

end = time.time()
print('total time spent {}'.format((end-start)/60))

ratio = int(len(ds)*.7)

train_ds = ds[1:ratio,:]
test_ds = ds[ratio:,:]

print(train_ds.shape, test_ds.shape)


# adds the gaussian noise based on the mean and the standard deviation 
# def add_gaussian_noise(data):
#   mean = (10, 10, 10)
#   std = (50, 50, 50)
#   row, col, channel = data.shape
#   noise = np.random.normal(mean, std, (row, col, channel)).astype('uint8')
#   return data + noise

# def add_gaussian_noise(data):
#   return data + np.random.normal(scale=0.05, size=data.shape)
  
# def add_gaussian_to_dataset(data):
#   count = 0
#   end = len(data)
#   output_data = []
#   while count < end:
#     output_data.append(add_gaussian_noise(data[count]))
#     count+=1
#   return np.array(output_data)

def add_gaussian_to_dataset(data):
    return data + np.random.normal(scale=0.0, size=data.shape)

train_ds = train_ds / 255
test_ds = test_ds / 255

gaussian_train_ds = add_gaussian_to_dataset(train_ds)
gaussian_test_ds = add_gaussian_to_dataset(test_ds)

#plt.imshow((gaussian_train_ds[1,:]*255).astype(np.uint8))

# https://www.tensorflow.org/tutorials/generative/autoencoder
# put the Activation layer AFTER the BatchNormalization() layer


class Autoencoder(tf.keras.Model):

  def __init__(self, dense_dim, latent_dim):
    super(Autoencoder, self).__init__()
    self.dense_dim = dense_dim
    self.latent_dim = latent_dim
  
    self.encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(width, height, 3)),
        
        tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPool2D(padding='same'),
          
        tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPool2D(padding='same'),          

        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPool2D(padding='same'),          
          
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(self.dense_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(self.latent_dim, activation='tanh'),
        tf.keras.layers.BatchNormalization(),

        ])
  
    self.decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),

        tf.keras.layers.Dense(self.dense_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(5*10*channels_, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Reshape(target_shape=(5, 10, channels_)),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.UpSampling2D(),

        tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.UpSampling2D(),
          
        tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.UpSampling2D(),
          
        tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.UpSampling2D(),
          
        tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation=None),

        ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    
    return decoded


# initialize the model
dense_dim = 1000
latent_dim = 100
channels_ = 64
dropout = 0.25

#opt = tf.keras.optimizers.Adam(learning_rate=0.002)
opt = keras.optimizers.Adamax(learning_rate=0.001)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    gaussian_auto_encoder = Autoencoder(dense_dim, latent_dim)
    gaussian_auto_encoder.compile(loss='mse', optimizer=opt)

# create checkpoint and save best weight
checkpoint_path = "/home/ryo/Program/Ashesh_colab/ae_cp/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


def train():

    # see model summary
    gaussian_auto_encoder.encoder.summary()
    gaussian_auto_encoder.decoder.summary()

    start = time.time()

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     mode='min',
                                                     save_best_only=True)

    # early stopping if not changing for 50 epochs
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=100)

    # reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.1,
                                                     patience=50, 
                                                     verbose=1,
                                                     min_lr=0.00001)

    # train the model
    gaussian_history = gaussian_auto_encoder.fit(gaussian_train_ds,
                                                 train_ds, 
                                                 epochs=500,
                                                 batch_size=64,
                                                 shuffle=True,
                                                 validation_data=(gaussian_test_ds, test_ds),
                                                 callbacks=[cp_callback, early_stop, reduce_lr])

    end = time.time()
    print('\ntotal time spent {}'.format((end-start)/60))

    plt.plot(gaussian_history.epoch, gaussian_history.history['loss'], label='train_loss')
    plt.plot(gaussian_history.epoch, gaussian_history.history['val_loss'], label='test_loss')
    plt.title('Epochs on Training Loss')
    plt.xlabel('# of Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    return gaussian_history


def test():
    # load_best_checkpoint and evaluate
    cp_model = Autoencoder(dense_dim, latent_dim)
    cp_model.compile(loss='mse', optimizer=opt)
    cp_model.load_weights(checkpoint_path)
    cp_loss = cp_model.evaluate(gaussian_test_ds, test_ds)

    # evaluate the model on the test set
    # final_loss = gaussian_auto_encoder.evaluate(gaussian_test_ds, test_ds)

    """# DENOISED IMAGES"""

    # run model on the test_ds to reconstruct
    cp_result = cp_model.predict(gaussian_test_ds)
    
    n = 10
    #idx = [np.random.randint(1,20) for i in range(n)]
    idx = [np.random.randint(1,20) for i in range(n)]
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.title("original")
        plt.imshow(test_ds[idx[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display original + noise
        bx = plt.subplot(3, n, i + n + 1)
        plt.title("original+noise")
        plt.imshow(gaussian_test_ds[idx[i]])
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)

        # display reconstruction
        cx = plt.subplot(3, n, i + 2*n + 1)
        plt.title("reconstructed")
        plt.imshow(cp_result[idx[i]])
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)

    plt.show()

    # final_result = gaussian_auto_encoder.predict(gaussian_test_ds)

    # samples = len(final_result)
    # fig = plt.figure(figsize=(15, samples))
    # for p in range(1, samples):
    #     ax = fig.add_subplot(samples//2, 5, p+1)
    #     ax.imshow((final_result[p]*255).astype(np.uint8))
    #     ax.axis('off')

    # """# Original Test Images"""
    
    # fig = plt.figure(figsize=(15, samples))
    # for p in range(1, samples):
    #     ax = fig.add_subplot(samples//2, 5, p+1)
    #     ax.imshow(test_ds[p])
    #     ax.axis('off')