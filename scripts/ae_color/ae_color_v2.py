import os, sys, glob, re, time, datetime
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
from keras.optimizers import Adam
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


def explore_ds(path, file_type):
  print(path) #print path
  os.chdir(path)
  ds = sorted(glob.glob(file_type), key=numerical_sort)
  # print(ds) #print contents
  return ds


def visualize_ds(images, vis=False):
  
  samples = len(images)
  # print('total images', samples)

  if vis:
    fig = plt.figure(figsize=(10,samples))
    fig.subplots_adjust(hspace=0.1)
  
  for p in range(samples):
    img = plt.imread(images[p])
    img = cv2.resize(img, (width, height))

    BATCH.append(img) #global variable

    if vis:
      ax = fig.add_subplot(samples//2, 4, p+1)
      ax.axis('off')
      ax.imshow(img)
    

def process_ds(dir_list):
  for group in dir_list:
    images = explore_ds(os.path.join(path, dir, group), '*.jpg')
    visualize_ds(images, vis=False)
    # break



# clone sample images
path = '/content/drive/MyDrive/ds/pushing'
sys.path.append(path)
dirs = os.listdir(path)

width = 300
height =300 

BATCH = []

start = time.time()
for dir in dirs:
  dir_list = explore_ds(os.path.join(path, dir), 'group*')
  process_ds(dir_list)

  # if dir=='05':
  break
  

end = time.time()
print('total time spent {}'.format((end-start)/60))

print(len(BATCH))

start = time.time()

ds = tf.stack(BATCH) #create tensor of samples

end = time.time()
print('total time spent {}'.format((end-start)/60))

