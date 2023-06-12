import os
import torch
# import torch.nn as nn
import argparse
import matplotlib.pylab as plt
# import matplotlib.animation as anim

from torchinfo import summary
from eipl_utils import tensor2numpy
from eipl_print_func import print_info
from eipl_arg_utils import restore_args
from KonbiniForceMapData import *
from force_estimation_v4 import ForceEstimationDINOv2, ForceEstimationResNet

import forcemap
from force_estimation import force_distribution_viewer
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
fmap = forcemap.GridForceMap('konbini_shelf')

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--idx',      type=str, default='0' )
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--datasize', type=str, default='1k')
args = parser.parse_args()

# check args
# if args.filename is None and not args.pretrained:
#     assert False, 'Please set filename or pretrained'

print_info("load pretrained weight")
# args.filename = os.path.join('log', '20230606_1017_32', 'CAE.pth' ) # 1k
args.filename = os.path.join('log', '20230606_1854_41', 'CAE.pth' ) # 4k resenc + resdec, compiled
# args.filename = os.path.join('log', '20230612_1057_39', 'CAE.pth' ) # 1k resenc + resdec

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args( os.path.join(dir_name, 'args.json') )
idx    = int(args.idx)

# load dataset
minmax = [params['vmin'], params['vmax']]
print('loading test data ...')
test_data  = KonbiniRandomSceneDataset('validation', minmax, datasize=args.datasize)
images, labels = test_data.get_data()
# images_raw, _ = test_data.get_raw_data()
# images_raw = images_raw.transpose(0,2,3,1)
# T = images.shape[1]


# define model
# model = ForceEstimationDINOv2(device='cpu')
model = ForceEstimationResNet(device='cpu')
print(summary(model, input_size=(32, 3, 336, 672)))

# load weight and compile
model = torch.compile(model)
# ckpt = torch.load(args.filename, map_location=torch.device('cpu'))
ckpt = torch.load(args.filename)
model.load_state_dict(ckpt['model_state_dict'])
model.to('cpu')
model.eval()

# prediction
batch = images[0:8]
_yi = model(batch)
yi = tensor2numpy(_yi)
yi = yi.transpose(0,2,3,1)

force_labels = tensor2numpy(labels[0:8]).transpose(0,2,3,1)


import pandas as pd

def load_bin_state(i):
    if test_data.datasize == '1k':
        start_index = 750
    elif test_data.datasize == '4k':
        start_index = 3000
    else:
        assert False, 'Unknowk dataset'
    return pd.read_pickle(f'/home/ryo/Dataset/dataset2/konbini-stacked/bin_state{i+start_index:05}.pkl')


def plot_forcemap(i):
    fmap.set_values(yi[i])
    # bin_state = self.test_data[2][n] if visualize_bin_state else None
    # viewer.publish_bin_state(bin_state, fmap)
    bin_state = load_bin_state(i)
    viewer.publish_bin_state(bin_state, fmap)


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
