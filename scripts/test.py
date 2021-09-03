# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import roi

height = 80
width = 160
crop_size = (40, 40)


def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, (height, width))

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


model = model()

imgs = np.array(roi.load_dataset(groups=[2])[0][1])
