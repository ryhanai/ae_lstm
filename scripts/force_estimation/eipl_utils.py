#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
# 

import os
import cv2
import glob
import datetime
import numpy as np
import matplotlib.pylab as plt


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data          


def deprocess_img(data, vmin=-0.9, vmax=0.9):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        vmin (float):  Minimum value of input data
        vmax (float):  Maximum value of input data
    Return:
        data (np.array with np.uint8): Normalized data array from 0 to 255.
    """
    data[np.where(data<vmin)] = vmin
    data[np.where(data>vmax)] = vmax
    return normalization(data, [vmin, vmax], [0,255]).astype(np.uint8)


def tensor2numpy(x):
    """
    Convert tensor to numpy array.
    """
    if x.device.type == 'cpu':
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()


def resize_img(img, size=(64,64), reshape_flag=True):
    """
    Convert tensor to numpy array.
    """
    if len(img.shape) == 5:
        N,T,W,H,C = img.shape
        img = img.reshape((-1,) + img.shape[2:])
    else:
        reshape_flag = False
        
    imgs = []
    for i in range(len(img)):
        imgs.append( cv2.resize(img[i], size) )

    imgs = np.array(imgs)
    if reshape_flag:
        imgs = imgs.reshape(N, T, size[1], size[0], 3)
    return imgs


def plt_img(img, key=None, title=None, dtype=None):
    """
    Convert tensor to numpy array.
    """
    width, height = img.shape

    if dtype is not None:
        plt.imshow( img.astype(dtype))
    else:
        plt.imshow( img )

    if key is not None:
        plt.plot(key[0]*width, key[1]*height, 'ro', markersize=3)

    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.title(title)
    plt.tick_params(length=0)


def check_path(path, mkdir=False):
    """
    checks given path is existing or not
    """
    if path[-1] == '/':
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.mkdir(path)
        else:
            raise ValueError("%s does not exist" % path)
    return path
                                                    

def set_logdir(log_dir, tag):
    return check_path(os.path.join(log_dir,tag), mkdir=True)


def calc_minmax(_data):
    data = _data.reshape(-1, _data.shape[-1])
    data_minmax = np.array( [ np.min(data,0),\
                              np.max(data,0) ] )
    return data_minmax


def list_to_numpy(data_list, max_N):
    dtype = data_list[0].dtype
    array = np.ones( (len(data_list), max_N,) + data_list[0].shape[1:], dtype)

    for i, data in enumerate(data_list):
        N = len(data)
        array[i, :N] = data[:N].astype(dtype)
        array[i, N:] = array[i, N:] * data[-1].astype(dtype)
        
    return array


def cos_interpolation(data, step=20):
    data = data.copy()
    points = np.diff(data)

    for i, p in enumerate(points):
        if p == 1:
            t = np.linspace(0.0, 1.0, step*2)
        elif p == -1:
            t = np.linspace(1.0, 0.0, step*2)
        else:
            continue
        
        x_latent = (1-np.cos(t*np.pi)) /2
        data[i-step+1:i+step+1] = x_latent
    
    return np.expand_dims( data, axis=-1 )