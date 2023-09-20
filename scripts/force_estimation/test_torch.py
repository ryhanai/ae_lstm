#
# Copyright (c) 2023 Ryo Hanai
# 

import os
import time
import argparse
import numpy as np
import pandas as pd
import json
import functools
import operator

import torch
from torchinfo import summary
from eipl_utils import tensor2numpy
from eipl_print_func import print_info

from KonbiniForceMapData import *
from SeriaBasketForceMapData import *
from force_estimation_v4 import *

import forcemap
from force_estimation import force_distribution_viewer

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='log/20230627_1730_52/CAE.pth')
parser.add_argument('--dataset_path', type=str, default='./basket-filling3-c-1k')
args = parser.parse_args()

# check args
# if args.filename is None and not args.pretrained:
#     assert False, 'Please set filename or pretrained'

# restore model parameters
with open(os.path.join(os.path.split(args.filename)[0], 'args.json'), 'r') as f:
    model_params = json.load(f)
with open(os.path.join(args.dataset_path, 'params.json'), 'r') as f:
    dataset_params = json.load(f)

print_info(f'loading pretrained weight [{args.filename}]')
ckpt = torch.load(args.filename)
# ckpt = torch.load(args.filename, map_location=torch.device('cpu'))

data_loader = dataset_params['data loader']
print_info(f'loading test data [{data_loader}]')
# test_data = globals()[data_loader]('test', view_index=0)
test_data = globals()[data_loader]('validation', view_index=0)

model_class = model_params['model']
print_info(f'build model and load weights [{model_class}]')
model = globals()[model_class]()
model.load_state_dict(ckpt['model_state_dict'])


class Tester:
    def __init__(self, test_data, model, dataset_params):
        self._device = 'cuda'
        self.test_data = test_data
        self._model = model
        self._model.to(self._device)
        self._model.eval()
        print(summary(self._model, input_size=(16, 3, 336, 672)))
        # self._model = torch.compile(self._model)
        forcemap_scene = dataset_params['forcemap']
        self._fmap = forcemap.GridForceMap(forcemap_scene)

        # if there exists a bin_state file in the directory, load the bin states.
        bs_file = os.path.join(self.test_data.root_dir, self.test_data.task_name, self.test_data.data_type, 'bin_state_bz2.pkl')
        if os.path.exists(bs_file):
            self._bin_states = self._bin_states = pd.read_pickle(bs_file, compression='bz2')
        self._draw_range = [0.5, 0.9]

    def predict(self, idx, view_idx=None, show_result=True, log_scale=True, show_bin_state=True):
        if view_idx is not None:
            x = self.test_data.get_specific_view_and_force(idx, view_idx)[0]
        else:
            x = self.test_data[idx][0]
        batch = torch.unsqueeze(x, 0).to(self._device)
        t1 = time.time()
        y = self._model(batch)
        t2 = time.time()
        print_info(f'inference: {t2-t1} [sec]')

        if type(y) == tuple:  # MVE model
            means, vars = y
            mean = means[0]
            var = vars[0]
        else:
            mean = y[0]

        def post_process(y, log_scale=True):
            y = tensor2numpy(y)
            y = y.transpose(1, 2, 0)
            if not log_scale:
                y = np.exp((y - 0.1) / 0.8) - 1.0
            return y

        if type(y) == tuple:
            return post_process(mean, log_scale=log_scale), post_process(var, log_scale=log_scale)
        else:
            return post_process(mean, log_scale=log_scale)

    def predict_with_multiple_views(self, idx, view_indices=range(3), ord=1, eps=1e-7):
        def error_fn(x, y):
            return (np.abs(x - y) ** ord).mean()

        ms, vs = zip(*[self.predict(idx, view_idx) for view_idx in view_indices])
        ws = [1 / (v + eps) for v in vs]
        total_weight = functools.reduce(operator.add, ws)
        w_average = functools.reduce(operator.add, [ws[i] * ms[i] for i in view_indices]) / total_weight
        average = functools.reduce(operator.add,  ms) / len(view_indices)

        label = tensor2numpy(tester.test_data[idx][1]).transpose(1, 2, 0)
        print(f'W_AVERAGE: {error_fn(label, w_average)}')
        print(f'AVERAGE: {error_fn(label, average)}')
        for i in view_indices:
            print(f'VIEW{i}: {error_fn(label, ms[i])}')
        return w_average, average, ms

    def show_mean(self, idx, view_idx=None, show_bin_state=True):
        y = self.predict(idx, view_idx)

        if type(y) == tuple:  # MVE
            y = y[0]

        if show_bin_state:
            bs = self.load_bin_state(idx)
        else:
            bs = None
        self.show_forcemap(y, bs)

    def show_variance(self, idx, view_idx=None, show_bin_state=True, scale=2):
        y = self.predict(idx, view_idx)

        assert type(y) == tuple and len(y) == 2, 'the model is not MVE model'
        y = y[1]

        if show_bin_state:
            bs = self.load_bin_state(idx)
        else:
            bs = None
        self.show_forcemap(y / np.max(y) * scale, bs)

    def show_prediction_error(self, idx, predicted_force, ord=1, scale=1.):
        def error_fn(x, y):
            return np.abs(x - y) ** ord

        force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
        bs = self.load_bin_state(idx)
        error = error_fn(force_label, predicted_force)
        self.show_forcemap(error * scale, bs)
        return error

    def label(self, idx, log_scale=True):
        force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
        if not log_scale:
            force_label = np.exp((force_label - 0.1) / 0.8) - 1.0
        return force_label

    def show_label(self, idx, log_scale=True):
        force_label = self.label(idx, log_scale=log_scale)
        bs = self.load_bin_state(idx)
        self.show_forcemap(force_label, bs)
        
    def load_bin_state(self, idx):
        return self._bin_states[idx]

    def show_forcemap(self, fmap_values, bin_state, draw_lower_bound=0.4):
        self._fmap.set_values(fmap_values)
        # bin_state = self.test_data[2][n] if visualize_bin_state else None
        # viewer.publish_bin_state(bin_state, fmap)
        viewer.publish_bin_state(bin_state, self._fmap, draw_range=self._draw_range)

    def set_draw_range(self, values):
        self._draw_range = values

    def predict_force_with_multiviews(self, idx):
        eps = 1e-8
        ys = [self.predict_force(idx, view, with_variance=True) for view in range(3)]
        return ys
        # a = 0
        # s = 0
        # for mean, var in ys:
        #     var = var + eps
        #     a = a + mean / var
        #     s = s + 
        # return a


tester = Tester(test_data, model, dataset_params)

# Predict force (mean)
# tester.predict_force(idx, view=None)
# tester.predict_force(idx, view=1)  # specify the view

# Predict force variance
# tester.predict_force_variance(idx, view=None)

# Predict with multi-views


# tester.predict_forcce_variance
# - Predict force prediction error
# tester.show_force_prediction_error(idx)
#

def err(idx, log_scale=True):
    y_pred = tester.predict(idx, log_scale=log_scale)
    f_label = tester.label(idx, log_scale=log_scale)
    return y_pred, f_label
    # return np.sum(np.abs(y_pred - f_label)) / np.sum(f_label)


def KL(p, q):
    if type(p) != torch.Tensor:
        p = torch.Tensor(p) 
    if type(q) != torch.Tensor:
        q = torch.Tensor(q)
    p = p / p.sum()
    q = q / q.sum()
    return (p * (p / q).log()).sum()

def f_aux(y, f, minmax):
    indices = np.where((f >= minmax[0]) & (f <= minmax[1]))
    if len(indices[0]) > 0:
        return len(indices[0]), np.average(np.abs(y[indices] - f[indices])), np.std(np.abs(y[indices] - f[indices]))
        # return len(indices[0]), y[indices], f[indices]
    else:
        return None
    # return len(indices[0]), np.average(np.abs(y[indices] - f[indices])), np.std(np.abs(y[indices] - f[indices]))


def f(predictions, minmax=[0.1, 0.2]):
    res = []
    for y, f  in predictions:
        r = f_aux(y, f, minmax)
        print(r)
        if r is not None:
            res.append(r)
    return res


import functools
import operator
import matplotlib

matplotlib.use('TkAgg')


def g(predictions, minmax=[0.1, 0.2]):
    n_samples, means, stds = zip(*f(predictions, minmax))
    return functools.reduce(operator.add, n_samples), np.average(list(filter(np.isfinite, means))), np.average(list(filter(np.isfinite, stds)))


def h(predictions):
    res = []
    for i in range(1, 8):
        minmax = [0.1 * i, 0.1 * (i + 1)]
        n_samples, means, stds = g(predictions, minmax)
        res.append((f'[{minmax[0]:.1f}, {minmax[1]:.1f}]', n_samples, means, stds))
    return res
