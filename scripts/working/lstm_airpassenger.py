# -*- coding: utf-8 -*-

import os, sys, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


filepath = 'AirPassengers.csv'
data = pd.read_csv(filepath)
data.head()

# 型変換
input_data = data['Passengers'].values.astype(float)
print("input_data : " , input_data.shape ,type(input_data))

# スケールの正規化
norm_scale = input_data.max()
input_data /= norm_scale
print(input_data[0:5])

# 入力データと教師データの作成
def make_dataset(low_data, maxlen):

    data, target = [], []

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target


# RNNへの入力データ数
window_size = 12

# 入力データと教師データへの分割
X, y = make_dataset(input_data, window_size)
print("shape X : " , X.shape)
print("shape y : " , y.shape)

# 訓練データ/検証データ分割



class JointLSTM(tf.keras.Model):

    def __init__(self, dof=10):
        super(JointLSTM, self).__init__()

        self._maxlen = 20
        self._dof = 1

        self.lstm = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, self._dof)),
            tf.keras.layers.Dense(self._dof)
        ])

    def call(self, x):
        return self.lstm(x)


opt = keras.optimizers.Adam()
lstm = JointLSTM()
lstm.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

def train():
    start = time.time()

    history = lstm.fit(train_data,
                       train_labels, 
                       epochs=50,
                       batch_size=20,
                       shuffle=True,
                       validation_data=(val_data, val_labels),
                       callbacks=[])
                       #callbacks=[cp_callback, early_stop, reduce_lr])

    end = time.time()
    print('\ntotal time spent {}'.format((end-start)/60))

    lstm.lstm.summary()
    
    plt.plot(history.epoch, history.history['loss'], label='train_loss')
    plt.plot(history.epoch, history.history['val_loss'], label='test_loss')
    plt.title('Epochs on Training Loss')
    plt.xlabel('# of Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    return history
