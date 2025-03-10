import numpy as np
import pandas as pd
# import cv2
import os


def normalization(data, indataRange, outdataRange):
    """     
    Function to normalize a numpy array within a specified range
    Args:   
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. ind    ataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indat    aRange=[0.0, 1.0].
    Return: 
        data (np.array): Normalized data array
    """     
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data


def decode_force(normalized_log_impulse, dt=1./60):
    bounds = np.load(os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling3-c-1k/force_bounds.npy'))
    log_impulse = normalization(normalized_log_impulse, [0.1, 0.9], bounds)
    impulse = np.exp(log_impulse) - 1
    force = impulse / dt # N
    return force


def post_process(force):
    """return gf

    Args:
        force (_type_): _description_

    Returns:
        _type_: _description_
    """
    g = 9.8
    dV = 0.19*0.26*0.20 / (40**3)
    return force * dV / g * 1e3


def load_forcemap(n):
    return pd.read_pickle(f'/Users/ryo/Downloads/20230720_weight_experiment/data/forcemap{n:05}.pkl')

# np.sum(np.where(a >= 0.6, a, 0))