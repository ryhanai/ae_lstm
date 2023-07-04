import os
import time
import argparse
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json

import torch
from torchinfo import summary
from eipl_utils import tensor2numpy
from eipl_print_func import print_info

from KonbiniForceMapData import *
from SeriaBasketForceMapData import *
from force_estimation_v4 import *

import forcemap
from force_estimation import force_distribution_viewer
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True


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
test_data = globals()[data_loader]('test')

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

    def predict_by_index(self, idx, show_result=True, log_scale=True, show_bin_state=True):
        batch = torch.unsqueeze(self.test_data[idx][0], 0).to(self._device)
        t1 = time.time()
        yi = self._model(batch)[0]
        t2 = time.time()
        print_info(f'inference took {t2-t1} [sec]')
        yi = tensor2numpy(yi)
        yi = yi.transpose(1, 2, 0)
        if not log_scale:
            yi = np.exp((yi - 0.1) / 0.8) - 1.0
        if show_result:
            if show_bin_state:
                bs = self.load_bin_state(idx)
            else:
                bs = None
            self.show_forcemap(yi, bs)
        return yi

    def predict_variance_by_index(self, idx, show_result=True, show_bin_state=True, ratio=2):
        batch = torch.unsqueeze(self.test_data[idx][0], 0).to(self._device)
        means, vars = self._model(batch)
        yi = vars[0]
        yi = tensor2numpy(yi)
        yi = yi.transpose(1, 2, 0)
        if show_result:
            if show_bin_state:
                bs = self.load_bin_state(idx)
            else:
                bs = None
            self.show_forcemap(yi/ np.max(yi) * ratio, bs)
        return yi

    def show_prediction_error(self, idx):
        force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
        batch = torch.unsqueeze(self.test_data[idx][0], 0).to(self._device)
        yi = self._model(batch)[0]
        yi = tensor2numpy(yi)
        yi = yi.transpose(1, 2, 0)
        bs = self.load_bin_state(idx)
        error = force_label - yi
        self.show_forcemap(np.abs(error), bs)
        return error

    def show_label_by_index(self, idx, log_scale=True):
        force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
        if not log_scale:
            force_label = np.exp((force_label - 0.1) / 0.8) - 1.0
        bs = self.load_bin_state(idx)
        self.show_forcemap(force_label, bs)
        return force_label

    def load_bin_state(self, idx):
        # if test_data.datasize == '1k':
        #     start_index = 750
        # elif test_data.datasize == '4k':
        #     start_index = 3000
        # else:
        #     assert False, 'Unknowk dataset'
        # return pd.read_pickle(os.path.join(os.environ['HOME'], f'Dataset/dataset2/konbini-stacked/bin_state{start_index+idx:05}.pkl'))

        start_index = 1750
        return pd.read_pickle(os.path.join(os.environ['HOME'], f'Dataset/dataset2/basket-filling3/bin_state{start_index+idx:05}.pkl'))

    def show_forcemap(self, fmap_values, bin_state):
        self._fmap.set_values(fmap_values)
        # bin_state = self.test_data[2][n] if visualize_bin_state else None
        # viewer.publish_bin_state(bin_state, fmap)
        viewer.publish_bin_state(bin_state, self._fmap)


tester = Tester(test_data, model, dataset_params)

# tester.predict(idx)
# tester.show_labels(idx)
