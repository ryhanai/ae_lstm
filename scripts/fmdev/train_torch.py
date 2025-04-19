#
# Copyright (c) 2023 Ryo Hanai
#

import argparse
import json
import random
from pathlib import Path
import re
import importlib
import numpy as np

from fmdev.eipl_arg_utils import check_args
from fmdev.eipl_print_func import print_info, print_error
from fmdev.eipl_utils import set_logdir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchinfo import summary

from tqdm import tqdm
import wandb


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        self.val_loss_min = np.inf

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


class PCLoss(nn.Module):
    def __init__(self, delta=0.01):
        super().__init__()
        self._delta = delta

    def forward(self, y_hat, y, sdf):
        weight = 1. / (torch.abs(sdf) / self._delta + 1.) ** 2
        loss = (weight * ((y_hat - y) ** 2)).mean()
        return loss


class Trainer:
    """
    Helper class to train convolutional neural network with datalodaer

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        device (str):
    """

    def __init__(self, model, optimizer, log_dir_path, loss="mse", device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self._log_dir_path = log_dir_path
        self._loss = loss

    def gen_chkpt_path(self, tag):
        return str(Path(self._log_dir_path) / f"{tag}.pth")

    def save(self, epoch, loss, tag=None):
        save_name = self.gen_chkpt_path(f'{epoch:05d}' if tag == None else 'best')
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            save_name,
        )

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()

        total_loss = 0.0

        assert self._loss == "mse" or self._loss == "pcl", f"Unknown loss function: {self._loss}"

        for n_batch, bi in enumerate(data):
            if self._loss == "mse":
                xi, yi = bi
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                yi_hat = self.model(xi)
                loss = nn.MSELoss()(yi_hat, yi)
                total_loss += loss.item()
            elif self._loss == "pcl":
                xi, yi, sdf = bi
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                sdf = sdf.to(self.device)
                yi_hat = self.model(xi)
                loss = PCLoss()(yi_hat, yi, sdf)
                total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / n_batch

class MVELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self._eps = 1e-12
        self._eps = 1e-6

    def forward(self, y_hat, y):
        mu_hat, sigma_hat = y_hat
        loss1 = torch.log(sigma_hat + self._eps)
        loss2 = (mu_hat - y) ** 2 / (sigma_hat + self._eps)
        loss3 = (mu_hat - y) ** 2
        l1m = loss1.mean()
        l2m = loss2.mean()
        l3m = loss3.mean()
        return l1m + l2m, l1m, l3m

        # loss1 = torch.log(sigma_hat + self._eps)
        # loss2 = (mu_hat - y) ** 2 / (sigma_hat + self._eps)
        # return loss2.mean(), loss1.mean()

        # loss1 = torch.log(sigma_hat + self._eps)
        # loss2 = nn.MSELoss()(mu_hat, y)
        return loss2, loss1.mean()


class TrainerMVE(Trainer):
    def __init__(self, model, optimizer, device="cpu"):
        super().__init__(model, optimizer, device=device)

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()

        total_loss = 0.0
        sig_term_loss = 0.0
        mu_term_loss = 0.0
        for n_batch, (xi, yi) in enumerate(data):
            xi = xi.to(self.device)
            yi = yi.to(self.device)

            y_hat = self.model(xi)
            loss, sig_term, mu_term = MVELoss()(y_hat, yi)
            total_loss += loss.item()
            sig_term_loss += sig_term.item()
            mu_term_loss += mu_term.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1), sig_term_loss / (n_batch + 1), mu_term_loss / (n_batch + 1)


# GPU optimizes and accelerates the network calculations.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# argument parser
parser = argparse.ArgumentParser(description="Learning to predict force distribution")
parser.add_argument("--dataset_path", type=str, default="~/Dataset/forcemap")
parser.add_argument("--task_name", type=str, default="tabletop240125")
parser.add_argument("--model", type=str)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--loss", type=str, default="mse")
parser.add_argument("--optimizer", type=str, default="adamax")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.1)
parser.add_argument("--vmax", type=float, default=0.9)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
parser.add_argument("--method", help="geometry-aware | isotropic | sdf", type=str, default="geometry-aware")
parser.add_argument("--sigma_f", type=float, default=0.03)
parser.add_argument("--sigma_g", type=float, default=0.01)
parser.add_argument("--pretrained_weights", help="use pre-trained weight", type=str, default="")
args = parser.parse_args()

# check args
args = check_args(args)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"


task_name=args.task_name

# load dataset
with open(Path(args.dataset_path) / task_name / "params.json", "r") as f:
    dataset_params = json.load(f)

data_loader = dataset_params["data loader"]
minmax = [args.vmin, args.vmax]

print_info(f"loading train data [{data_loader}]")
dataset_module = importlib.import_module('fmdev.TabletopForceMapData')
train_data = getattr(dataset_module, data_loader)("train", 
                                                  minmax, 
                                                  task_name=task_name, 
                                                  method=args.method,
                                                  sigma_f=args.sigma_f,
                                                  sigma_g=args.sigma_g,
                                                  load_sdf=args.loss == "pcl",
                                                  )

print_info(f"loading validation data [{data_loader}]")
valid_data = getattr(dataset_module, data_loader)("validation", 
                                                  minmax, 
                                                  task_name=task_name, 
                                                  method=args.method,
                                                  sigma_f=args.sigma_f,
                                                  sigma_g=args.sigma_g,
                                                  load_sdf=args.loss == "pcl",
                                                  )

train_sampler = BatchSampler(RandomSampler(train_data), batch_size=args.batch_size, drop_last=False)
train_loader = DataLoader(train_data, batch_size=None, num_workers=8, pin_memory=True, sampler=train_sampler)
valid_sampler = BatchSampler(RandomSampler(valid_data), batch_size=args.batch_size, drop_last=False)
valid_loader = DataLoader(valid_data, batch_size=None, num_workers=8, pin_memory=True, sampler=valid_sampler)


set_seed_everywhere(args.seed)

mod_name = re.sub('\\.[^.]+$', '', args.model)
model_module = importlib.import_module(mod_name)
model_class_name = re.sub('[^.]*\\.', '', args.model)
model = getattr(model_module, model_class_name)(fine_tune_encoder=True, device=args.device)

# print_info('Check the effect of seed:')
# print(model.decoder.state_dict()['0.conv1.weight'][0][0])

if args.pretrained_weights != "":
    print_error('loading pretrained weight is not supported yet')
    # with open(Path(args.pretrained_weights) / "args.json", "r") as f:
    #     model_params = json.load(f)

    # assert args.method == model_params["method"]

    # weight_file = Path(args.pretrained_weights) / f"{model.__class__.__name__}.pth"
    # print_info(f"load pre-trained weights from '{weight_file}'")
    # ckpt = torch.load(weight_file)
    # model.load_state_dict(ckpt["model_state_dict"])

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "adamax":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr, eps=1e-4)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam or Adamax.".format(args.optimizer)


log_dir_path = set_logdir("./" + args.log_dir, args.tag)
trainer = Trainer(model, optimizer, log_dir_path=log_dir_path, loss=args.loss, device=device)
# summary(trainer.model, input_size=(1, 3, 360, 512))
early_stop = EarlyStopping(patience=100000)


# loss, sig_term, mu_term = trainer.process_epoch(test_loader, training=False)
# print_info(f'Initialized model performance (test_loss): {loss}/{sig_term}/{mu_term}')


# def initialization_test(n_times, mve=True):
#     for n in range(n_times):
#         if mve:
#             model = ForceEstimationResNetSeriaBasketMVE(mean_network_weights, device=args.device)
#             optimizer = optim.RAdam(model.parameters(), lr=args.lr, eps=1e-4)
#             trainer = TrainerMVE(model, optimizer, device=device)
#             loss, sig_term, mu_term = trainer.process_epoch(test_loader, training=False)
#             print_info(f"Initialized model performance (test_loss/sig_term/mu_term): {loss}/{sig_term}/{mu_term}")
#         else:
#             model = ForceEstimationResNetSeriaBasket(fine_tune_encoder=True, device=args.device)
#             model.load_state_dict(mean_network_weights)
#             optimizer = optim.RAdam(model.parameters(), lr=args.lr, eps=1e-4)
#             trainer = Trainer(model, optimizer, device=device)
#             print_info(
#                 f"Initialized model performance (train_loss/test_loss): {trainer.process_epoch(train_loader, training=False)}/{trainer.process_epoch(test_loader, training=False)}"
#             )


def do_train():
    config = args.__dict__
    config['dataset_class'] = type(valid_data)

    if model_class_name == 'ForceEstimationV5':
        model_tag = 'transformer'
    elif model_class_name == 'ForceEstimationResNetSeriaBasket':
        model_tag = 'resnet'
    
    if config['method'] == 'isotropic':
        group = f"IFS_f{config['sigma_f']:.3f}_{model_tag}"
        name = f"IFS_f{config['sigma_f']:.3f}_{config['tag']}_{model_tag}"
    if config['method'] == 'geometry-aware':
        group = f"GAFS_f{config['sigma_f']:.3f}_g{config['sigma_g']:.3f}_{model_tag}"
        name = f"GAFS_f{config['sigma_f']:.3f}_g{config['sigma_g']:.3f}_{config['tag']}_{model_tag}"
    wandb.init(project="forcemap", group=group, name=name, config=config)

    with tqdm(range(args.epoch)) as pbar_epoch:
        for epoch in pbar_epoch:
            train_loss = trainer.process_epoch(train_loader)
            test_loss = trainer.process_epoch(valid_loader, training=False)

            wandb.log({"Loss/train_loss": train_loss, "Loss/test_loss": test_loss})

            # early stop
            save_ckpt, stop_ckpt = early_stop(test_loss)

            if save_ckpt:
                trainer.save(epoch, [train_loss, test_loss], 'best')
            else:
                if epoch % 50 == 49:
                    trainer.save(epoch, [train_loss, test_loss])

            # print process bar
            postfix = f"train_loss={train_loss:.5e}, test_loss={test_loss:.5e}"
            pbar_epoch.set_postfix_str(postfix)

    wandb.finish()


if __name__ == '__main__':
    do_train()
