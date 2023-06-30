import os
import time
import torch
# import torch.nn as nn
import argparse
import matplotlib.pylab as plt
# import matplotlib.animation as anim
import numpy as np
import pandas as pd

from torchinfo import summary
from eipl_utils import tensor2numpy
from eipl_print_func import print_info
from eipl_arg_utils import restore_args
from KonbiniForceMapData import *
from SeriaBasketForceMapData import *
from force_estimation_v4 import ForceEstimationResNetSeriaBasket

import forcemap
from force_estimation import force_distribution_viewer
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
# parser.add_argument('--idx',      type=str, default='0' )
parser.add_argument('--dataset_path', type=str, default='./basket-filling3-c-1k')
args = parser.parse_args()

# check args
# if args.filename is None and not args.pretrained:
#     assert False, 'Please set filename or pretrained'

print_info("load pretrained weight")
# args.filename = os.path.join('log', '20230606_1017_32', 'CAE.pth' ) # 1k
# args.filename = os.path.join('log', '20230606_1854_41', 'CAE.pth' ) # 4k resenc + resdec, compiled
# args.filename = os.path.join('log', '20230612_1057_39', 'CAE.pth' ) # 1k resenc + resdec
# args.filename = os.path.join('log', '20230612_1914_05', 'CAE.pth' ) # 1k resenc + resdec, dino fine-tune
# args.filename = os.path.join('log', '20230612_1601_40', 'CAE.pth' ) # 1k resenc + resdec, no fine-tune
args.filename = os.path.join('log', '20230627_1730_52', 'CAE.pth' ) # basket-filling

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args( os.path.join(dir_name, 'args.json') )
# idx    = int(args.idx)

ckpt = torch.load(args.filename)
# ckpt = torch.load(args.filename, map_location=torch.device('cpu'))


dataset_class = pd.read_pickle(os.path.join(args.dataset_path, 'params.pkl'))['dataset class']
print_info('loading test data ...')
test_data = globals()[dataset_class]('validation')
print_info('build model and load weights ...')
model_class = 'ForceEstimationResNetSeriaBasket'  # this name is given by 'args.json'
model = globals()[model_class](device='cpu')  # for fast inference, use GPU
model.load_state_dict(ckpt['model_state_dict'])


class Tester:
    def __init__(self, test_data, model, params):
        self._device = 'cuda'
        self.test_data = test_data
        self._model = model
        self._model.to(self._device)
        self._model.eval()
        print(summary(self._model, input_size=(16, 3, 336, 672)))
        # load weight and compile
        # self._model = torch.compile(self._model)

        forcemap_scene = 'seria_basket'  # this is identified by dataset
        self._fmap = forcemap.GridForceMap(forcemap_scene)

        # minmax = [params['vmin'], params['vmax']]

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

        start_index = 1500
        return pd.read_pickle(os.path.join(os.environ['HOME'], f'Dataset/dataset2/basket-filling3/bin_state{start_index+idx:05}.pkl'))

    def show_forcemap(self, fmap_values, bin_state):
        self._fmap.set_values(fmap_values)
        # bin_state = self.test_data[2][n] if visualize_bin_state else None
        # viewer.publish_bin_state(bin_state, fmap)
        viewer.publish_bin_state(bin_state, self._fmap)


tester = Tester(test_data, model, params)

# tester.predict(idx)
# tester.show_labels(idx)

# model = ForceEstimationDINOv2(device='cpu')
# model = ForceEstimationResNet(device='cpu')

def plot_forcelabel(i):
    fmap.set_values(force_labels[i])
    bin_state = load_bin_state(i)
    viewer.publish_bin_state(bin_state, fmap)


# fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=60)
# def anim_update(i):
#     for j in range(2):
#         ax[j].cla()

#     ax[0].imshow(images_raw[idx,i,:,:,::-1])
#     ax[0].axis('off')
#     ax[0].set_title('Input image')

#     ax[1].imshow(yi[i,:,:,::-1])
#     ax[1].axis('off')
#     ax[1].set_title('Reconstructed image')

# ani = anim.FuncAnimation(fig, anim_update, interval=int(T/10), frames=T)
# ani.save( './output/{}_{}.gif'.format(params['model'], idx) )
