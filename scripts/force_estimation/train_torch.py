#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
# 

import os
import sys
import torch
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
# from eipl.model import BasicCAE, CAE, BasicCAEBN, CAEBN
# from eipl.data import GraspBottleImageDataset
from eipl_utils import set_logdir
from eipl_arg_utils import check_args


import torch
import torch.nn as nn

from KonbiniForceMapData import *
from force_estimation_v4 import ForceEstimationDINOv2


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience  (int):  Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.save_ckpt = False
        self.stop_flag = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss):

        if np.isnan(val_loss) or np.isinf(val_loss):
            raise RuntimeError("Invalid loss, terminating training") 
        
        score = -val_loss

        if self.best_score is None:
            self.save_ckpt = True
            self.best_score = score
        elif score < self.best_score:
            self.save_ckpt = False
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.save_ckpt = True
            self.best_score = score
            self.counter = 0

        return self.save_ckpt, self.stop_flag        


class Trainer:
    """
    Helper class to train convolutional neural network with datalodaer

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        batch_size (int): 
        stdev (float): 
        device (str): 
    """
    def __init__(self,
                model,
                optimizer,
                device='cpu'):

        self.device = device
        self.optimizer = optimizer        
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def process_epoch(self, data, training=True):
        
        if not training:
            self.model.eval()

        total_loss = 0.0
        for n_batch, (xi, yi) in enumerate(data):
            xi = xi.to(self.device)
            yi = yi.to(self.device)

            yi_hat = self.model(xi)
            loss = nn.MSELoss()(yi_hat, yi)
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / n_batch


# GPU optimizes and accelerates the network calculations.
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# argument parser
parser = argparse.ArgumentParser(description='Learning convolutional autoencoder')
parser.add_argument('--model',       type=str, default='CAE'    )
parser.add_argument('--epoch',       type=int, default=10000    )
parser.add_argument('--batch_size',  type=int, default=32       )
parser.add_argument('--feat_dim',    type=int, default=10       )
parser.add_argument('--stdev',       type=float, default=0.02   )
parser.add_argument('--lr',          type=float, default=1e-3   )
parser.add_argument('--optimizer',   type=str,   default='adam' )
parser.add_argument('--log_dir',     default='log/'             )
parser.add_argument('--vmin',        type=float, default=0.1    )
parser.add_argument('--vmax',        type=float, default=0.9    )
parser.add_argument('--device',      type=int,   default=0      )
parser.add_argument('--tag',         help='Tag name for snap/log sub directory')
args = parser.parse_args()

# check args
args = check_args(args)

# set device id
if args.device >= 0:
    device = 'cuda:{}'.format(args.device)
else:
    device = 'cpu'

# load dataset
minmax = [args.vmin, args.vmax]
train_data = KonbiniRandomSceneDataset('train', minmax, stdev=args.stdev)
test_data  = KonbiniRandomSceneDataset('validation', minmax, stdev=args.stdev)

train_sampler = BatchSampler(
    RandomSampler(train_data),
    batch_size=args.batch_size,
    drop_last=False)

train_loader = DataLoader(
    train_data,
    batch_size=None,
    num_workers=8,
    pin_memory=True,
    sampler=train_sampler)

test_sampler = BatchSampler(
    RandomSampler(test_data),
    batch_size=args.batch_size,
    drop_last=False)

test_loader = DataLoader(
    train_data,
    batch_size=None,
    num_workers=8,
    pin_memory=True,
    sampler=test_sampler)

# define model
model = ForceEstimationDINOv2(device=args.device)


# torch compile for pytorch 2.0
if int(torch.__version__[0]) >= 2:
    model = torch.compile(model)

# set optimizer
if args.optimizer.casefold() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == 'radam':
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, 'Unknown optimizer name {}. please set Adam or RAdam.'.format(args.optimizer)

# load trainer/tester class
trainer = Trainer( model, optimizer, device=device )

### training main
log_dir_path = set_logdir('./'+args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, '{}.pth'.format(args.model) )
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=100000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        test_loss  = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar('Loss/train_loss', train_loss, epoch)
        writer.add_scalar('Loss/test_loss',  test_loss,  epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name )

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss,
                                            test_loss=test_loss))