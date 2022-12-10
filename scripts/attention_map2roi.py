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


def detect_rect_region(x, filter_size=(3, 3), alpha=0.0):
    a = np.average(np.average(x, axis=1), axis=1)
    a = np.expand_dims(a, (1, 2))
    a = np.tile(a, (1, 20, 40, 1))
    x = x - a - alpha
    W = tf.ones((filter_size[0], filter_size[1], 1, 1))
    # W = tf.ones(1, (filter_size[0], filter_size[1], 1))  # <- This would be correct
    scores = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return scores

# kxs,kys = np.meshgrid(range(3,13),range(3,13))
# kernel_sizes = list(zip(kxs.ravel(), kys.ravel()))


def apply_filters(images,
                  attention_map,
                  alpha=0.0,
                  beta=4.0,
                  use_softmax=True,
                  visualize_result=True,
                  return_result=False):

    N, H, W, C = attention_map.shape

    if use_softmax:
        a = attention_map.reshape(N, 20*40)
        a = tf.nn.softmax(a*beta)
        attention_map = np.reshape(a, attention_map.shape)

    kxs = list(range(3, 13))
    kys = list(range(3, 13))
    score = np.empty((len(kys), len(kxs), N, H, W, C))

    for i, ky in enumerate(kys):
        for j, kx in enumerate(kxs):
            kernel_sz = (ky, kx)
            score[i, j] = detect_rect_region(attention_map, kernel_sz, alpha)  # (N,H,W,C=1)
    score = np.transpose(score, [2, 0, 1, 3, 4, 5])  # move batch dimesion to the head

    cs = []
    ss = []

    for sc in score:  # loop over images
        density = sc[:, :, :, :, 0]
        sy, sx, y, x = add_offsets(density)  # pixel coords to (center,scale) representation
        c = [y/20., x/40.]
        s = [(3+sy)/20., (3+sx)/40.]
        cs.append(c)
        ss.append(s)

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


# def apply_filters(images, attention_map, alpha=0.0, beta=4.0, use_softmax=True, visualize_result=True, return_result=False):
#     n_imgs = attention_map.shape[0]

#     if use_softmax:
#         a = attention_map.reshape(n_imgs, 20*40)
#         a2 = tf.nn.softmax(a*beta)
#         a3 = np.reshape(a2, attention_map.shape)
#         attention_map = a3
    
#     score = np.array([detect_rect_region(attention_map, kernel_sz, alpha) for kernel_sz in kernel_sizes])
#     cs = []
#     ss = []
#     for img_idx in range(n_imgs):
#         idx = np.unravel_index(np.argmax(score[:,img_idx]), score[:,img_idx].shape)
#         c,s = idx2rect(idx)
#         cs.append(c)
#         ss.append(s)

#     b = draw_bounding_boxes(images, roi_rect1((np.array(cs), np.array(ss))))

#     if visualize_result:
#         visualize_ds(attention_map)
#         visualize_ds(b)
#         plt.show()

#     if return_result:
#         return b
 
