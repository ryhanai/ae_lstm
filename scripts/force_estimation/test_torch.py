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

import numpy as np
import torch
import torch._dynamo
from app.pick_planning import LiftingDirectionPlanner
from core.object_loader import ObjectInfo
from force_estimation import force_distribution_viewer, forcemap
from force_estimation.eipl_print_func import print_info
from force_estimation.eipl_utils import tensor2numpy
from force_estimation.force_estimation_v4 import *
from force_estimation.force_estimation_v5 import *

# from KonbiniForceMapData import *
# from SeriaBasketForceMapData import *
from force_estimation.TabletopForceMapData import *
from torchsummary import summary

torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
viewer.set_object_info(ObjectInfo(dataset="ycb_conveni_v1", split="train"))


parser = argparse.ArgumentParser()
# parser.add_argument("--filename", type=str, default="log/20230627_1730_52/CAE.pth")
# parser.add_argument("--dataset_path", type=str, default="./basket-filling3-c-1k")
parser.add_argument("--dataset_path", type=str, default=f"{os.environ['HOME']}/Dataset/forcemap")
parser.add_argument("--task_name", type=str, default="tabletop240125")
parser.add_argument("--data_split", type=str, default="test")
# parser.add_argument("--weights", type=str, default="log/20240221_0015_58 log/20240227_1431_21 log/20240301_1431_11")
# parser.add_argument("--weights", type=str, default="log/20240221_0015_58 log/20240304_1834_24 log/20240304_2000_11")
parser.add_argument(
    "--weights",
    type=str,
    default=f"{os.environ['HOME']}/Program/moonshot/ae_lstm/scripts/force_estimation/log/20240221_0015_58 {os.environ['HOME']}/Program/moonshot/ae_lstm/scripts/force_estimation/log/20240304_1834_24",
)
parser.add_argument("--weight_file", type=str, default="best.pth")
args = parser.parse_args()


# previous result
# geometry-guided: "log/20240130_1947_53"
# isotropic: "log/20240201_1847_01"

# mesh2sdf result
# geometry-guided: "log/20240221_0015_58
# isotropic: "log/20240221_1851_57"
# SDF: "log/20240222_1649_39"


def setup_dataloader(args):
    root_dir = Path(args.dataset_path)
    task_name = args.task_name

    with open(os.path.join(args.dataset_path, task_name, "params.json"), "r") as f:
        dataset_params = json.load(f)

    data_loader = dataset_params["data loader"]
    print_info(f"loading test data [{data_loader}], split={args.data_split}")
    test_data = globals()[data_loader](args.data_split, root_dir=root_dir, task_name=task_name)
    return test_data, dataset_params


weight_dirs = args.weights.split()

test_data, dataset_params = setup_dataloader(args)
fmap = forcemap.GridForceMap(dataset_params["forcemap"])
planner = LiftingDirectionPlanner(fmap)


class Tester:
    def __init__(self, test_data, weight_dirs, fmap, planner=None):
        self._device = "cuda"
        self.test_data = test_data

        self._models = {}
        for weight_dir in weight_dirs:
            self.setup_model(weight_dir)

        self._fmap = fmap
        self._planner = planner
        self._draw_range = [0.4, 0.9]

    def setup_model(self, weight_dir):
        with open(Path(weight_dir) / "args.json", "r") as f:
            model_params = json.load(f)

        model_class = model_params["model"]
        # weight_file = f"{model_file}/{model_class}.pth"
        weight_file = f"{weight_dir}/{args.weight_file}"
        print_info(f"loading pretrained weight [{weight_file}]")
        ckpt = torch.load(f"{weight_file}")
        # ckpt = torch.load(args.weights, map_location=torch.device('cpu'))

        print_info(f"building model [{model_class}]")
        model = globals()[model_class]()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        # model = torch.compile(model)
        summary(model, input_size=(3, 360, 512))
        self._models[model_params["method"]] = model

    def predict_from_image_file(
        self, image_file, log_scale=True, planning=True, object_radius=0.05, show_result=True, visualize_idx=1
    ):
        img = cv2.imread(image_file)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict_from_image(img, log_scale, planning, object_radius, show_result, visualize_idx)

    def crop_center_d415(self, img, c=(20, 10), crop=64):
        return img[180 + c[0] : 540 + c[0], 320 + c[1] + crop : 960 + c[1] - crop]

    def predict_from_image(
        self, rgb_image, log_scale=True, planning=True, object_radius=0.05, show_result=True, visualize_idx=1
    ):
        roi = self.crop_center_d415(rgb_image)
        roi = roi.transpose(2, 0, 1).astype("float32")
        roi = normalization(roi, (0.0, 255.0), [0.1, 0.9])
        x_batch = np.expand_dims(roi, axis=0)
        x_batch = torch.from_numpy(x_batch).float()
        result = self.do_predict(x_batch, log_scale=True, object_radius=0.05)
        if show_result:
            self.show_result(None, *result, show_bin_state=False, visualize_idx=visualize_idx)
        return result

    def predict(
        self,
        idx,
        view_idx=0,
        log_scale=True,
        downstream_task=True,
        object_radius=0.05,
        show_result=True,
        visualize="geometry-aware",
    ):
        # prepare data
        x_batch, f_batch = self.test_data.__getitem__([idx], view_idx)
        predicted_maps = self.do_predict(x_batch, log_scale=True)

        # viewer.rviz_client.delete_all()

        if show_result:
            self.show_result(idx, predicted_maps, visualize=visualize)

        if downstream_task:
            downstream_task_results = self.perform_down_stream_task(predicted_maps, object_radius=object_radius)

        return predicted_maps, downstream_task_results

    def perform_down_stream_task(self, predicted_maps, object_radius=0.05):
        planning_results = {}
        object_center = viewer.rviz_client.getObjectPosition()
        print_info(f"object center: {object_center}")
        force_bounds = self.test_data._compute_force_bounds()

        planning_results = {}
        for tag, y in predicted_maps.items():
            predicted_force_map = np.exp(normalization(y, test_data.minmax, np.log(force_bounds)))
            print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")
            v_omega = self._planner.pick_direction_plan(predicted_force_map, object_center, object_radius=object_radius)
            print_info(f"planning result: {v_omega[0]}, {v_omega[1]}")
            planning_results[tag] = v_omega

        # draw lifting directions
        object_center = viewer.rviz_client.getObjectPosition()
        arrow_scale = [0.005, 0.01, 0.004]
        arrow_colors = {"isotropic": [1, 0, 1, 1], "geometry-aware": [1, 1, 0, 1], "sdf": [0, 1, 1, 1]}
        for tag, v_omega in planning_results.items():
            lift_direction = v_omega[0]
            planner.draw_result(viewer, object_center, lift_direction, rgba=arrow_colors[tag], arrow_scale=arrow_scale)
        viewer.rviz_client.show()

        return planning_results

    def do_predict(self, x_batch, log_scale=True):
        def post_process(y, log_scale=True):
            y = tensor2numpy(y)
            y = y.transpose(1, 2, 0)
            if not log_scale:
                y = np.exp((y - 0.1) / 0.8) - 1.0
            return y

        x_batch = x_batch.to(self._device)

        predicted_maps = {}
        for tag, m in self._models.items():
            t1 = time.time()
            y = m(x_batch)[0]
            y = post_process(y, log_scale=log_scale)
            t2 = time.time()
            print_info(f"inference: {t2-t1} [sec]")
            predicted_maps[tag] = y

        return predicted_maps

    def show_result(
        self,
        idx,
        predicted_maps={},
        show_bin_state=True,
        visualize="geometry-aware",
    ):
        if show_bin_state:
            bs = self.test_data.load_bin_state(idx)
        else:
            bs = None

        self.show_forcemap(predicted_maps[visualize], bs, draw_range=self._draw_range)
        viewer.rviz_client.show()

    def show_force_label(self, idx, method="geometry-aware"):
        if method == "geometry-aware":
            method_num = 1
        elif method == "isotropic":
            method_num = 0
        elif method == "sdf":
            method_num = 2

        m = self.test_data._method  # save self.test_data._method
        self.test_data._method = method_num
        f = self.test_data.load_fmap(idx)
        self.test_data._method = m  # restore self.test_data._method
        f = f.transpose(1, 2, 0)
        bs = self.test_data.load_bin_state(idx)
        self.show_forcemap(f, bs, draw_range=self._draw_range)
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

    def get_draw_range(self):
        return self._draw_range

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


tester = Tester(test_data, weight_dirs, fmap, planner)


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


# import operator

# import matplotlib

# matplotlib.use("TkAgg")


# def g(predictions, minmax=[0.1, 0.2]):
#     n_samples, means, stds = zip(*f(predictions, minmax))
#     return (
#         functools.reduce(operator.add, n_samples),
#         np.average(list(filter(np.isfinite, means))),
#         np.average(list(filter(np.isfinite, stds))),
#     )


# def h(predictions):
#     res = []
#     for i in range(1, 8):
#         minmax = [0.1 * i, 0.1 * (i + 1)]
#         n_samples, means, stds = g(predictions, minmax)
#         res.append((f"[{minmax[0]:.1f}, {minmax[1]:.1f}]", n_samples, means, stds))
#     return res
