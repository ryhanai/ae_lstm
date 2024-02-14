#
# Copyright (c) 2023 Ryo Hanai
#

import argparse
import functools
import json
import operator
import os
import time
from pathlib import Path

import forcemap
import numpy as np
import pandas as pd
import torch
import torch._dynamo
from app.pick_planning import LiftingDirectionPlanner
from eipl_print_func import print_info
from eipl_utils import tensor2numpy
from force_estimation import force_distribution_viewer
from force_estimation_v4 import *

# from KonbiniForceMapData import *
# from SeriaBasketForceMapData import *
from TabletopForceMapData import *
from torchinfo import summary

torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

parser = argparse.ArgumentParser()

# parser.add_argument("--filename", type=str, default="log/20230627_1730_52/CAE.pth")
# parser.add_argument("--dataset_path", type=str, default="./basket-filling3-c-1k")

data_dir = f"{os.environ['HOME']}/Dataset/forcemap/tabletop240125"
parser.add_argument("--dataset_path", type=str, default=data_dir)
parser.add_argument("--weights", type=str, default="log/20240130_1947_53")  # geometry-guided model
parser.add_argument("--baseline_weights", type=str, default="log/20240201_1847_01")  # iso-tropic model
args = parser.parse_args()


def setup_model(weights):
    with open(Path(weights) / "args.json", "r") as f:
        model_params = json.load(f)

    model_class = model_params["model"]
    weight_file = f"{weights}/{model_class}.pth"
    print_info(f"loading pretrained weight [{weight_file}]")
    ckpt = torch.load(f"{weight_file}")
    # ckpt = torch.load(args.weights, map_location=torch.device('cpu'))

    print_info(f"building model [{model_class}]")
    model = globals()[model_class]()
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def setup_dataloader(args):
    with open(os.path.join(args.dataset_path, "params.json"), "r") as f:
        dataset_params = json.load(f)

    data_loader = dataset_params["data loader"]
    print_info(f"loading test data [{data_loader}]")
    test_data = globals()[data_loader]("test")
    # test_data = globals()[data_loader]("validation", view_index=0)
    return test_data, dataset_params


model = setup_model(args.weights)

if args.baseline_weights != "":
    baseline_models = [setup_model(args.baseline_weights)]
else:
    baseline_models = []

test_data, dataset_params = setup_dataloader(args)
fmap = forcemap.GridForceMap(dataset_params["forcemap"])
planner = LiftingDirectionPlanner(fmap)


class Tester:
    def __init__(self, test_data, model, fmap, planner=None, baseline_models=[]):
        self._device = "cuda"
        self.test_data = test_data

        self._model = model
        self._model.to(self._device)
        self._model.eval()

        self._baseline_models = []
        for bm in baseline_models:
            bm.to(self._device)
            bm.eval()
            self._baseline_models.append(bm)

        print(summary(self._model, input_size=(16, 3, 336, 672)))
        # self._model = torch.compile(self._model)
        self._fmap = fmap
        self._planner = planner
        self._draw_range = [0.5, 0.9]

    def load_bin_state(self, idx):
        p = Path(self.test_data.root_dir) / self.test_data.task_name / f"bin_state{idx:05d}.pkl"
        if os.path.exists(p):
            bs = pd.read_pickle(p)
        return bs

    def predict(self, idx, view_idx=0, log_scale=True, planning=True, object_radius=0.05):
        def post_process(y, log_scale=True):
            y = tensor2numpy(y)
            y = y.transpose(1, 2, 0)
            if not log_scale:
                y = np.exp((y - 0.1) / 0.8) - 1.0
            return y

        # prepare data
        x_batch, f_batch = self.test_data.__getitem__([idx], view_idx)
        x_batch = x_batch.to(self._device)

        # force prediction
        t1 = time.time()
        y = self._model(x_batch)[0]
        t2 = time.time()
        print_info(f"inference: {t2-t1} [sec]")
        y = post_process(y, log_scale=log_scale)

        # force prediction using baseline models
        bm_results = []
        for bm in self._baseline_models:
            yb = bm(x_batch)[0]
            yb = post_process(yb, log_scale=log_scale)
            bm_results.append(yb)

        # lifting direction planning
        if planning:
            object_center = viewer.rviz_client.getObjectPosition()
            print_info(f"object center: {object_center}")
            force_bounds = self.test_data._compute_force_bounds()

            predicted_force_map = np.exp(normalization(y, test_data.minmax, np.log(force_bounds)))
            print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")
            v_omega = self._planner.pick_direction_plan(predicted_force_map, object_center, object_radius=object_radius)
            print_info(f"planning result: {v_omega[0]}, {v_omega[1]}")
            planning_result = v_omega

            bm_planning_results = []
            for yb in bm_results:
                print_info("BASELINES:")
                predicted_force_map = np.exp(normalization(yb, test_data.minmax, np.log(force_bounds)))
                print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")
                v_omega = self._planner.pick_direction_plan(
                    predicted_force_map, object_center, object_radius=object_radius
                )
                print_info(f"planning result: {v_omega[0]}, {v_omega[1]}")
                bm_planning_results.append(v_omega)

        return idx, y, object_center, planning_result, bm_results, bm_planning_results

    def show_result(
        self,
        idx,
        y,
        object_center=None,
        planning_result=None,
        bm_results=[],
        bm_planning_results=[],
        show_bin_state=True,
    ):
        if show_bin_state:
            bs = self.load_bin_state(idx)
        else:
            bs = None
        self.show_forcemap(y, bs, draw_range=self._draw_range)

        arrow_scale = [0.005, 0.01, 0.004]
        if planning_result:
            pick_direction = planning_result[0]
            planner.draw_result(viewer, object_center, pick_direction, rgba=[1, 0, 1, 1], arrow_scale=arrow_scale)
        for bm_planning_result in bm_planning_results:
            pick_direction = bm_planning_result[0]
            planner.draw_result(viewer, object_center, pick_direction, rgba=[1, 1, 0, 1], arrow_scale=arrow_scale)

        viewer.rviz_client.show()

    # def predict_with_multiple_views(self, idx, view_indices=range(3), ord=1, eps=1e-7):
    #     def error_fn(x, y):
    #         return (np.abs(x - y) ** ord).mean()

    #     ms, vs = zip(*[self.predict(idx, view_idx) for view_idx in view_indices])
    #     ws = [1 / (v + eps) for v in vs]
    #     total_weight = functools.reduce(operator.add, ws)
    #     w_average = functools.reduce(operator.add, [ws[i] * ms[i] for i in view_indices]) / total_weight
    #     average = functools.reduce(operator.add, ms) / len(view_indices)

    #     label = tensor2numpy(tester.test_data[idx][1]).transpose(1, 2, 0)
    #     print(f"W_AVERAGE: {error_fn(label, w_average)}")
    #     print(f"AVERAGE: {error_fn(label, average)}")
    #     for i in view_indices:
    #         print(f"VIEW{i}: {error_fn(label, ms[i])}")
    #     return w_average, average, ms

    # def show_mean(self, idx, view_idx=0, log_scale=True, show_bin_state=True, draw_range=[0.4, 0.9], planning=True):
    #     y = self.predict(idx, view_idx, log_scale)

    #     if type(y) == tuple:  # MVE
    #         y = y[0]

    #     if show_bin_state:
    #         bs = self.load_bin_state(idx)
    #     else:
    #         bs = None
    #     self.show_forcemap(y, bs, draw_range=draw_range)
    #     if planning:
    #         # predicted_force_map = y
    #         force_bounds = self.test_data._compute_force_bounds()
    #         predicted_force_map = np.exp(normalization(y, test_data.minmax, np.log(force_bounds)))
    #         print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")

    #         object_center = viewer.rviz_client.getObjectPosition()
    #         print_info(f"object center: {object_center}")

    #         v, omega = self._planner.pick_direction_plan(
    #             predicted_force_map, object_center, object_radius=0.05, viewer=viewer
    #         )
    #         print_info(f"planning result: {v}, {omega}")

    # def show_variance(self, idx, view_idx=None, show_bin_state=True, scale=2):
    #     y = self.predict(idx, view_idx)

    #     assert type(y) == tuple and len(y) == 2, "the model is not MVE model"
    #     y = y[1]

    #     if show_bin_state:
    #         bs = self.load_bin_state(idx)
    #     else:
    #         bs = None
    #     self.show_forcemap(y / np.max(y) * scale, bs)

    # def show_prediction_error(self, idx, predicted_force, ord=1, scale=1.0):
    #     def error_fn(x, y):
    #         return np.abs(x - y) ** ord

    #     force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
    #     bs = self.load_bin_state(idx)
    #     error = error_fn(force_label, predicted_force)
    #     self.show_forcemap(error * scale, bs)
    #     return error

    def label(self, idx, log_scale=True):
        force_label = tensor2numpy(self.test_data[idx][1]).transpose(1, 2, 0)
        if not log_scale:
            force_label = np.exp((force_label - 0.1) / 0.8) - 1.0
        return force_label

    def show_label(self, idx, log_scale=True):
        force_label = self.label(idx, log_scale=log_scale)
        bs = self.load_bin_state(idx)
        self.show_forcemap(force_label, bs)

    def show_forcemap(self, fmap_values, bin_state, draw_range=[0.4, 0.9]):
        self._fmap.set_values(fmap_values)
        viewer.publish_bin_state(bin_state, self._fmap, draw_range=draw_range)

    def set_draw_range(self, values):
        self._draw_range = values

    # def predict_force_with_multiviews(self, idx):
    #     eps = 1e-8
    #     ys = [self.predict_force(idx, view, with_variance=True) for view in range(3)]
    #     return ys
    #     # a = 0
    #     # s = 0
    #     # for mean, var in ys:
    #     #     var = var + eps
    #     #     a = a + mean / var
    #     #     s = s +
    #     # return a


tester = Tester(test_data, model, fmap, planner, baseline_models=baseline_models)


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
    for y, f in predictions:
        r = f_aux(y, f, minmax)
        print(r)
        if r is not None:
            res.append(r)
    return res


import functools
import operator

import matplotlib

matplotlib.use("TkAgg")


def g(predictions, minmax=[0.1, 0.2]):
    n_samples, means, stds = zip(*f(predictions, minmax))
    return (
        functools.reduce(operator.add, n_samples),
        np.average(list(filter(np.isfinite, means))),
        np.average(list(filter(np.isfinite, stds))),
    )


def h(predictions):
    res = []
    for i in range(1, 8):
        minmax = [0.1 * i, 0.1 * (i + 1)]
        n_samples, means, stds = g(predictions, minmax)
        res.append((f"[{minmax[0]:.1f}, {minmax[1]:.1f}]", n_samples, means, stds))
    return res
