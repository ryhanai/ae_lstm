# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import roi
from utils import *

roi_shape = (40, 80, 3)
roi_size = roi_shape[:2]
input_image_shape = (90, 160, 3)
input_image_size = input_image_shape[:2]

def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, roi_size)

def model_crop_and_resize():
    image_input = tf.keras.Input(shape=input_image_shape)
    roi_input = tf.keras.Input(shape=(4, ))
    cropped = tf.keras.layers.Lambda(crop_and_resize, output_shape=roi_size) ([image_input, roi_input])
    return tf.keras.Model([image_input, roi_input], cropped, name='test')

def test_crop_and_resize():
    model = model_crop_and_resize()
    imgs = np.array(load_dataset(groups=[2])[0][1]) / 255
    rois = np.tile([0.2, 0.2, 0.8, 0.8], [imgs.shape[0], 1])
    result = model([imgs, rois])
    plt.imshow(result[50])
    plt.show()

class GeneratorTest:
    def __init__(self):
        self.batch_size = batch_size
        self.time_window = time_window_size
        train_ds = load_dataset(groups=range(1,50))
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(train_ds, const_roi_fun, batch_size=self.batch_size, time_window_size=self.time_window, add_roi=False)
    def batch(self, n=10):
        start = time.time()
        for i in range(n):
            batch = next(self.train_gen)
        end = time.time()
        print('\ntook {}[sec]'.format((end-start)/n))

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


def model():
    image_input = tf.keras.Input(shape=(270, 480, 3))
    roi_input = tf.keras.Input(shape=(4, ))

    cropped = tf.keras.layers.Lambda(crop_and_resize, output_shape=(height, width))([image_input, roi_input])
    
    #cropped = tf.keras.layers.Lambda(lambda x: tf.image.crop_and_resize(x, roi_input, box_indices, crop_size), output_shape=crop_size) (image_input)

    #cropped = tf.keras.layers.Lambda(lambda x: tf.image.crop_and_resize(x[0], x[1], box_indices, crop_size), output_shape=crop_size) ([image_input], roi_input)
    
    return tf.keras.Model([image_input, roi_input], cropped, name='test')


# def crop_and_resize(args):
#     images, rois = args
#     #x,y,w,h = roi # This doesn't work
#     res = tf.zeros((images.shape[0], height, width, images.shape[3]))
#     for i in images.shape[0]:
#         y = rois[i][0]
#         x = rois[i][1]
#         h = rois[i][2]
#         w = rois[i][3]
#         cropped = tf.image.crop_to_bounding_box(images, y, x, h, w)
#         resized = tf.image.resize(cropped, (height, width), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False)
#         res[i] = resized
#     return res

# def model():
#     image_input = tf.keras.Input(shape=(height, width, 3))
#     roi_input = tf.keras.Input(shape=(4, ))
#     #roi_input = tf.constant([20, 50, 40, 40])

#     cropped = tf.keras.layers.Lambda(crop_and_resize, output_shape=crop_size)([image_input, roi_input])
    
#     return tf.keras.Model([image_input, roi_input], cropped, name='test')


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


model = model()

imgs = np.array(roi.load_dataset(groups=[2])[0][1])


# x,y,t,_ = trainer.test()
# def f(i):
#     env.moveArm(unnormalize_joint_position(x[i][-1])[:-1])
#     sync()
#     ximg = getImg()
#     env.moveArm(unnormalize_joint_position(t[i])[:-1])
#     sync()
#     yimg = getImg()

#     fig = plt.figure(figsize=(10,samples))
#     fig.subplots_adjust(hspace=0.1)

#     for p in range(samples):
#         ax = fig.add_subplot(samples//4, 4, p+1)
#         ax.axis('off')
#         ax.imshow(images[p])
#         if len(rois) > 0:
#             roi = rois[samples][0]
#             height = images[p].shape[0]
#             width = images[p].shape[1]
#             x = width * roi[1]
#             w = width * (roi[3] - roi[1])
#             y = height * roi[0]
#             h = height * (roi[2] - roi[0])
#             rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='red', fill=False) # x,y,w,h [pixels]
#             ax.add_patch(rect)


# from PIL import Image

# def overlay(img1, img2):
#     img1 = img1 * 255
#     im1 = Image.fromarray(img1.astype(np.uint8))
#     img2 = img2 * 255
#     im2 = Image.fromarray(img2.astype(np.uint8))
#     im2.putalpha(128)
#     merged = Image.new('RGBA', (160,90), (0,0,0,0))
#     merged.paste(im1, (0,0), None)
#     merged.paste(im2, (0,0), None)
#     plt.imshow(merged)
#     plt.show()
#     return merged
