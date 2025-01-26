#
# Copyright (c) 2023 Ryo Hanai
#

import argparse
import json
import os
import time
import random
from pathlib import Path

import numpy as np

# from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from eipl_arg_utils import check_args
from eipl_print_func import print_info
from eipl_utils import set_logdir
from force_estimation_v4 import *
from force_estimation_v5 import *

# from KonbiniForceMapData import *
# from SeriaBasketForceMapData import *
from TabletopForceMapData import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

#　from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

from torchsummary import summary
from tqdm import tqdm


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
        device (str):
    """

    def __init__(self, model, optimizer, log_dir_path, device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.model = model.to(self.device)
        summary(self.model, input_size=(3, 360, 512))
        self._log_dir_path = log_dir_path

    def gen_chkpt_path(self, tag):
        return os.path.join(self._log_dir_path, f"{tag}.pth")

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
parser = argparse.ArgumentParser(description="Learning convolutional autoencoder")
parser.add_argument("--dataset_path", type=str, default="~/Dataset/forcemap")
parser.add_argument("--task_name", type=str, default="tabletop240125")
parser.add_argument("--model", type=str, default="ForceEstimationResNetTabletop")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stdev", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adamax")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.1)
parser.add_argument("--vmax", type=float, default=0.9)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
parser.add_argument("--method", help="geometry-aware | isotropic | sdf", type=str, default="geometry-aware")
parser.add_argument("--weights", help="use pre-trained weight", type=str, default="")
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
with open(os.path.join(args.dataset_path, task_name, "params.json"), "r") as f:
    dataset_params = json.load(f)

data_loader = dataset_params["data loader"]
print_info(f"loading test data [{data_loader}]")


minmax = [args.vmin, args.vmax]
print("loading train data ... ", end="")
t_start = time.time()
train_data = globals()[data_loader]("train", minmax, task_name=task_name, method=args.method)
print(f"{time.time() - t_start} [sec]")

print("loading validation data ... ", end="")
t_start = time.time()
test_data = globals()[data_loader]("validation", minmax, task_name=task_name, method=args.method)
print(f"{time.time() - t_start} [sec]")

train_sampler = BatchSampler(RandomSampler(train_data), batch_size=args.batch_size, drop_last=False)
train_loader = DataLoader(train_data, batch_size=None, num_workers=8, pin_memory=True, sampler=train_sampler)
test_sampler = BatchSampler(RandomSampler(test_data), batch_size=args.batch_size, drop_last=False)
test_loader = DataLoader(test_data, batch_size=None, num_workers=8, pin_memory=True, sampler=test_sampler)

# define model

# mean_network_weights = torch.load('log/20230627_1730_52/CAE.pth')['model_state_dict']
# model = ForceEstimationResNetSeriaBasketMVE(mean_network_weights, device=args.device)


# model_class = globals()[args.model]()
# print_info(f"Model: {model_class}")
# model = model_class(fine_tune_encoder=True, device=args.device)

# model.load_state_dict(mean_network_weights)

# model = ForceEstimationResNet(fine_tune_encoder=True, device=args.device)
# model = ForceEstimationResNetMVE(fine_tune_encoder=True, device=args.device)

# model = ForceEstimationDINOv2(device=args.device)
# model = ForceEstimationDinoRes(fine_tune_encoder=True, device=args.device)


model = globals()[args.model](fine_tune_encoder=True, device=args.device)

if args.weights != "":
    with open(Path(args.weights) / "args.json", "r") as f:
        model_params = json.load(f)

    assert args.method == model_params["method"]

    weight_file = Path(args.weights) / f"{model.__class__.__name__}.pth"
    print_info(f"load pre-trained weights from '{weight_file}'")
    ckpt = torch.load(weight_file)
    model.load_state_dict(ckpt["model_state_dict"])


# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "adamax":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr, eps=1e-4)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam or Adamax.".format(args.optimizer)


# load trainer/tester class
log_dir_path = set_logdir("./" + args.log_dir, args.tag)


trainer = Trainer(model, optimizer, log_dir_path=log_dir_path, device=device)
# trainer = TrainerMVE(model, optimizer, device=device)

# tensorboard
# writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)

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

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def do_train():
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project="forcemap",
        name=f"fmap_{time_stamp}",
        config={
            "model": type(model),
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "method": args.method,
            "pre_trained_weights": args.weights,
            "dataset_class": type(test_data),
            "dataset": task_name,
            })

    with tqdm(range(args.epoch)) as pbar_epoch:
        for epoch in pbar_epoch:
            # train and test
            # train_loss, train_sig_loss, train_mu_loss = trainer.process_epoch(train_loader)
            # test_loss, test_sig_loss, test_mu_loss  = trainer.process_epoch(test_loader, training=False)
            train_loss = trainer.process_epoch(train_loader)
            test_loss = trainer.process_epoch(test_loader, training=False)

            # tensorboard
            # writer.add_scalar("Loss/train_loss", train_loss, epoch)
            # writer.add_scalar("Loss/test_loss", test_loss, epoch)

            wandb.log({"Loss/train_loss": train_loss, "Loss/test_loss": test_loss})

            # early stop
            save_ckpt, stop_ckpt = early_stop(test_loss)

            if save_ckpt:
                trainer.save(epoch, [train_loss, test_loss], 'best')
            else:
                if epoch % 20 == 0:
                    trainer.save(epoch, [train_loss, test_loss])

            # print process bar
            # postfix = f'train_loss={train_loss:.4e}, test_loss={test_loss:.4e}, train_sig={train_sig_loss:.4e}, test_sig={test_sig_loss:.4e}, train_mu={train_mu_loss:.4e}, test_mu={test_mu_loss:.4e}'
            postfix = f"train_loss={train_loss:.5e}, test_loss={test_loss:.5e}"
            pbar_epoch.set_postfix_str(postfix)
            # pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))

    wandb.finish()


set_seed_everywhere(args.seed)
do_train()
