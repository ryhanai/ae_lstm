# -*- coding: utf-8 -*-

# import os
# import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from core.utils import *


def roi_rect1(args):
    c, s = args
    lt = (1-s) * c
    rb = (1-s) * c + s
    return tf.concat([lt, rb], axis=1)


# def spatial_soft_argmax(features):
#     # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
#     # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
#     # jointly over the image dimensions. 
#     features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [N * C, H * W])
#     softmax = tf.nn.softmax(features)
#     # Reshape and transpose back to original format.
#     softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])

#     # Assume that image_coords is a tensor of size [H, W, 2] representing the image
#     # coordinates of each pixel.
#     # Convert softmax to shape [N, H, W, C, 1]
#     softmax = tf.expand_dims(softmax, -1)
#     # Convert image coords to shape [H, W, 1, 2]
#     image_coords = tf.expand_dims(image_coords, 2)
#     # Multiply (with broadcasting) and reduce over image dimensions to get the result
#     # of shape [N, C, 2]
#     ss_argmax = tf.reduce_sum(softmax * image_coords, reduction_indices=[1, 2])
#     return ss_argmax


image_coords = tf.constant([(sy, sx, y, x) for sy in range(12) for sx in range(12) for y in range(20) for x in range(40)], dtype=tf.float64)


def spatial_soft_argmax(features):
    SY, SX, Y, X = features.shape
    features = tf.expand_dims(tf.reshape(features, SY*SX*Y*X), 0)
    softmax = tf.nn.softmax(features)
    softmax = tf.reshape(softmax, (SY*SX*Y*X, 1))
    # softmax = tf.cast(softmax, tf.float32)
    ss_argmax = tf.reduce_sum(softmax * image_coords, axis=0)
    return softmax, ss_argmax


# def detect_rect_region(x, filter_size=(3, 3), n_sigma=1.0):
#     mu = np.average(x, axis=(1,2))
#     sigma = np.std(x, axis=(1,2))
#     mu = np.expand_dims(mu, (1, 2))
#     sigma = np.expand_dims(sigma, (1, 2))
#     x = x - mu - n_sigma * sigma
#     x = np.where(x > 0., x, 0.)
#     W = tf.ones((filter_size[0], filter_size[1], 1, 1))
#     scores = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#     return scores


def detect_rect_region(x, W, n_sigma=1.0, epsilon=1e-3):
    mu = tf.reduce_mean(x, (1, 2))
    mu = tf.expand_dims(tf.expand_dims(mu, 1), 2)
    # sigma = tf.math.reduce_std(x, (1, 2))
    # sigma = tf.expand_dims(tf.expand_dims(sigma, 1), 2)
    # x = x - mu - n_sigma * sigma
    # x = tf.where(x > 0., x, 0.)

    x = x - mu
    x = tf.where(x > -epsilon, x, -epsilon)

    scores = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # zero-padding
    return scores


def gaussian_filter(filter_size):
    ky, kx = filter_size
    y = np.arange(-(ky-1)/2, (ky-1)/2+1, 1)
    x = np.arange(-(kx-1)/2, (kx-1)/2+1, 1)
    yy, xx = np.meshgrid(y, x)
    W = np.exp(-((2*yy/ky)**2 + (2*xx/kx)**2))
    W = np.expand_dims(W, (2,3))
    return W


filters = []
YMAX = 15
XMAX = 15
kys = list(range(3, YMAX))
kxs = list(range(3, XMAX))
for i, ky in enumerate(kys):
    for j, kx in enumerate(kxs):
        # W = tf.ones((ky, kx, 1, 1))
        W = gaussian_filter((ky, kx))
        filters.append((i, j, W))


def compute_score(attention_map, n_sigma=1.0, epsilon=1e-3):
    attention_map[:, 0] = 0.0

    N, H, W, C = attention_map.shape
    score = np.empty((YMAX-3, XMAX-3, N, H, W, C))

    for i, j, filter in filters:
        score[i, j] = detect_rect_region(attention_map, filter, n_sigma, epsilon)  # (N,H,W,C=1)
    score = np.transpose(score, [2, 0, 1, 3, 4, 5])  # move batch dimesion to the head
    return score


def apply_filters(images,
                  attention_map,
                  n_sigma=1.0,
                  epsilon=1e-3,
                  beta=4.0,
                  use_softmax=False,
                  visualize_result=True,
                  return_result=False):

    N, H, W, C = attention_map.shape

    if use_softmax:
        a = attention_map.reshape(N, 20*40)
        a = tf.nn.softmax(a*beta)
        attention_map = np.reshape(a, attention_map.shape)

    score = compute_score(attention_map, n_sigma, epsilon=epsilon)

    cs = []
    ss = []
    for sc in score:  # loop over images
        density = sc[:, :, :, :, 0]

        # ssa = spatial_soft_argmax(density)
        # print('SSA =', ssa)
        # sy, sx, y, x = ssa

        sy, sx, y, x = add_offsets(density)  # pixel coords to (center,scale) representation

        c = [y/20., x/40.]
        s = [(3+sy)/20., (3+sx)/40.]
        cs.append(c)
        ss.append(s)

    print(cs, ss)    
    rect = roi_rect1((np.array(cs), np.array(ss)))
    b = draw_bounding_boxes(images, rect)

    if visualize_result:
        visualize_ds(attention_map)
        visualize_ds(b)
        plt.show()

    if return_result:
        return b, rect


def compute_offset(density, max_coords, d=[1, 0, 0, 0]):
    ky, kx, y, x = max_coords
    KH, KW, H, W = density.shape
    f1 = density[(ky+d[0]) % KH, (kx+d[1]) % KW, (y+d[2]) % H, (x+d[3]) % W] - density[ky, kx, y, x]
    f2 = density[(ky-d[0]) % KH, (kx-d[1]) % KW, (y-d[2]) % H, (x-d[3]) % W] - density[ky, kx, y, x]
    if np.isclose(f2+f1, 0, atol=1e-3):
        offset = 0.0
    else:
        offset = 0.5 * (f2-f1)/(f2+f1)
    return offset


def add_offsets(density):
    max_coords = np.array(np.unravel_index(np.argmax(density), density.shape))
    return max_coords + np.array(list(map(lambda d: compute_offset(density, max_coords, d), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])))


def test(tr):
    xs, y_pred = tr.predict_images(return_data=True)
    xs = xs[0][:, -1]
    attention_maps = y_pred[2]
    a = attention_maps[:, -1]
    return apply_filters(xs, a, visualize_result=True)
