#
# Copyright (c) 2023 Ryo Hanai
#

import argparse
import json
import re
import time
import cv2
from pathlib import Path
import numpy as np
import importlib

import torch
import torch._dynamo
from torchsummary import summary

from force_estimation import force_distribution_viewer
from force_estimation.pick_planning import LiftingDirectionPlanner

from dataset.object_loader import ObjectInfo
from fmdev.eipl_print_func import print_info, print_error
from fmdev.eipl_utils import tensor2numpy, normalization
from fmdev import forcemap


torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
viewer.set_object_info(ObjectInfo(dataset="ycb_conveni_v1", split="train"))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="~/Dataset/forcemap")
parser.add_argument("--task_name", type=str, default="tabletop240304")
parser.add_argument("--data_split", type=str, default="test")
parser.add_argument(
    "--weight_files",
    type=str,
    default="log/20250318_1221_24/00080.pth",
)
args = parser.parse_args()


def setup_model(weight_file, device, show_model_summary=False):
    """
    Create a model class and load weigts from the specified weight file.
    This also returns the parameters used for the training.

    Args:
        weight_file
        device
    Returns:
        model: model initialized with the specified weight data
        model_params
    """
    weight_dir = Path(weight_file).parent
    with open(weight_dir / "args.json", "r") as f:
        model_params = json.load(f)

    model_class_path = model_params["model"]
    print_info(f"building model [{model_class_path}]")
    mod_name = re.sub('\.[^\.]+$', '', model_class_path)
    model_module = importlib.import_module(mod_name)
    model_class_name = re.sub('^.*\.', '', model_class_path)
    model = getattr(model_module, model_class_name)(initialize_encoder_with_pretrained_weight=False)

    print_info(f"loading pretrained weight [{weight_file}]")
    ckpt = torch.load(f"{weight_file}")
    # ckpt = torch.load(f"{weight_file}", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model_state_dict"])
    # model = torch.compile(model)    
    model.to(device)
    model.eval()
    if show_model_summary:
        summary(model, input_size=(3, 360, 512))
    return model, model_params


def setup_dataloader(dataset_path, task_name, data_split, model_params):
    """
    Args:
    
    Returns:
        datast (Dataset)
        forcemap (forcemap.ForceMap)
    """
    root_dir = Path(dataset_path).expanduser()
    with open(root_dir / task_name / "params.json", "r") as f:
        dataset_params = json.load(f)

    data_loader_class_name = dataset_params["data loader"]
    print_info(f"loading test data [{data_loader_class_name}], split={data_split}")
    data_loader_module = importlib.import_module('fmdev.TabletopForceMapData')
    dataset = getattr(data_loader_module, data_loader_class_name)(data_split,
                                                                  root_dir=root_dir,
                                                                  task_name=task_name,
                                                                  method=model_params['method'],
                                                                  sigma_f=model_params['sigma_f'],
                                                                  sigma_g=model_params['sigma_g'],
                                                                  )
    return dataset, forcemap.GridForceMap(dataset_params["forcemap"])


class Tester:
    def __init__(self, dataset_path, task_name, weight_files, data_split='test'):
        self._device = "cuda"

        self._model_dataset_pairs = []
        self._model_params = []

        for weight_file in weight_files:
            model, model_params = setup_model(weight_file, self._device)
            test_data, fmap = setup_dataloader(dataset_path, task_name, data_split, model_params)
            self._model_dataset_pairs.append((model, test_data))
            self._model_params.append(model_params)

        self._fmap = fmap
        self._draw_range = [0.4, 0.9]

    def predict(
        self,
        scene_idx,
        view_idx=0,
        log_scale=True,
        show_result=True,
        show_bin_state=True,
        visualize_idx=0,
    ):
        """
        log_scale: return results in log scale
        visualise_idx: choose one of the predicted forcemaps
        """
        predicted_maps = []
        for model, test_data in self._model_dataset_pairs:
            x_batch, f_batch = test_data.__getitem__([scene_idx], view_idx)
            predicted_maps.append(self.do_predict(model, x_batch, log_scale=log_scale))

        if show_result:
            bin_state_idx = None
            if show_bin_state:
                bin_state_idx = scene_idx
            self.show_result(predicted_maps[visualize_idx], bin_state_idx)

        return predicted_maps

    def do_predict(self, model, x_batch, log_scale=True):
        def post_process(y):
            y = tensor2numpy(y)
            y = y.transpose(1, 2, 0)
            if not log_scale:
                y = np.exp((y - 0.1) / 0.8) - 1.0
            return y

        x_batch = x_batch.to(self._device)
        t1 = time.time()
        y = model(x_batch)[0]
        y = post_process(y)
        t2 = time.time()
        print_info(f"inference: {t2-t1} [sec]")

        return y

    def show_result(self, fmap_values, bin_state_idx=None):
        if bin_state_idx == None:
            bs = None
        else:
            # It is assumed that all the dataset has the same bin_state
            bs = self._model_dataset_pairs[0][1].load_bin_state(bin_state_idx)

        self._fmap.set_values(fmap_values)
        viewer.clear()
        viewer.draw_bin_state(bs, self._fmap, draw_range=self._draw_range)
        viewer.show()

    def show_force_label(self, scene_idx, visualize_idx=0):
        model, dataset = self._model_dataset_pairs[visualize_idx]
        f = dataset.load_fmap(scene_idx)
        f = f.transpose(1, 2, 0)
        bs = dataset.load_bin_state(scene_idx)

        self._fmap.set_values(f)
        viewer.clear()
        viewer.draw_bin_state(bs, self._fmap, draw_range=self._draw_range)
        viewer.show()

        return f

    def predict_from_image(
        self,
        image,
        log_scale=True,
        show_result=True,
        visualize_idx=0,
    ):
        image = image.transpose(2, 0, 1).astype("float32")
        image = normalization(image, (0.0, 255.0), [0.1, 0.9])
        x_batch = np.expand_dims(image, axis=0)
        x_batch = torch.from_numpy(x_batch).float()

        predicted_maps = []
        for model, _ in self._model_dataset_pairs:
            predicted_maps.append(self.do_predict(model, x_batch, log_scale=log_scale))
        
        if show_result:
            self.show_result(predicted_maps[visualize_idx])

        return predicted_maps

    def predict_from_image_file(
        self,
        image_file_name,
        log_scale=True,
        show_result=True,
        visualize_idx=0,
    ):
        img = cv2.imread(image_file_name)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.predict_from_image(rgb_img, log_scale, show_result, visualize_idx)

    def crop_center_d415(self, img, c=(20, 10), crop=64):
        return img[180 + c[0] : 540 + c[0], 320 + c[1] + crop : 960 + c[1] - crop]

    def set_draw_range(self, values):
        self._draw_range = values

    def get_draw_range(self):
        return self._draw_range


class TesterWithLiftingPlanning(Tester):
    def __init__(self, 
                 dataset_path,
                 task_name,
                 weight_files,
                 data_split='test'
                 ):
        super().__init__(dataset_path, task_name, weight_files, data_split=data_split)
        self._planner = LiftingDirectionPlanner(self._fmap)
        self._arrow_scale = [0.005, 0.01, 0.004]
        self._arrow_colors_method = {"isotropic": [1., 0., 1., 1.], 
                                    "geometry-aware": [1., 1., 0., 1.], 
                                    "sdf": [0., 1., 1., 1.]}
        self._arrow_colors_index = [
                [1., 0.64705882, 0., 1.], # orange
                [0.50196078, 0., 0.50196078, 1.],  # purple
                [0., 1., 1., 1.],  # cyan
                [0.19607843, 0.80392157, 0.19607843, 1.],  # lime green
            ]

    def predict(self, 
                scene_idx,
                view_idx=0,
                log_scale=True,
                show_result=True,
                show_bin_state=True,
                visualize_idx=0,
                object_radius=0.05):

        predicted_maps = super().predict(scene_idx, view_idx, log_scale, show_result=False)

        object_center = viewer.rviz_client.getInteractiveMarkerPose()[0]
        planning_results = self.plan_lifting(predicted_maps, object_center, object_radius=object_radius)

        if show_result:
            bin_state_idx = None
            if show_bin_state:
                bin_state_idx = scene_idx
            self.show_result(predicted_maps[visualize_idx], planning_results, object_center, bin_state_idx)

        return predicted_maps, planning_results

    def predict_from_image(
        self,
        image,
        object_center,        
        log_scale=True,
        show_result=True,
        visualize_idx=0,
        object_radius=0.05,
    ):

        predicted_maps = super().predict_from_image(image, log_scale, show_result, visualize_idx)
        planning_results = self.plan_lifting(predicted_maps, object_center, object_radius=object_radius)

        if show_result:
            bin_state_idx = None
            self.show_result(predicted_maps[visualize_idx], planning_results, object_center, bin_state_idx)

        return predicted_maps, planning_results

    def show_result(self, fmap_values, planning_results, object_center, bin_state_idx=None):
        if bin_state_idx == None:
            bs = None
        else:
            # It is assumed that all the dataset has the same bin_state
            bs = self._model_dataset_pairs[0][1].load_bin_state(bin_state_idx)

        self._fmap.set_values(fmap_values)
        viewer.clear()
        viewer.draw_bin_state(bs, self._fmap, draw_range=self._draw_range)

        # draw planning results
        for i in range(len(self._model_dataset_pairs)):
            direction = planning_results[i]
            c = self.get_arrow_color(i)
            self._planner.draw_result(viewer,
                                      object_center,
                                      direction,
                                      rgba=c,
                                      arrow_scale=self._arrow_scale
                                     )

        viewer.show()

    def get_arrow_color(self, key):
        if type(key) == str:
            return self._arrow_colors_method[key]
        if type(key) == int:
            return self._arrow_colors_index[key]
        else:
            print_error('cannot get arrow color (unknown key)')

    def plan_lifting(self, predicted_maps, object_center, object_radius=0.05):        
        print_info(f"object center: {object_center}")

        planning_results = []
        for predicted_map, (model, dataset) in zip(predicted_maps, self._model_dataset_pairs):
            force_bounds = dataset._compute_force_bounds()
            # predicted_force_map = np.exp(normalization(predicted_map, dataset.minmax, np.log(force_bounds)))
            predicted_force_map = normalization(predicted_map, dataset.minmax, np.log(force_bounds))
            print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")
            v_omega = self._planner.pick_direction_plan(predicted_force_map, object_center, object_radius=object_radius)
            print_info(f"planning result [V, omega]: {v_omega[0]}, {v_omega[1]}")
            planning_results.append(v_omega[0])  # lifting direction

        return planning_results


if __name__ == '__main__':
    tester = TesterWithLiftingPlanning(args.dataset_path,
    # tester = Tester(args.dataset_path,    
                                    args.task_name,
                                    args.weight_files.split(),
                                    data_split=args.data_split)


# def err(idx, log_scale=True):
#     y_pred = tester.predict(idx, log_scale=log_scale)
#     f_label = tester.label(idx, log_scale=log_scale)
#     return y_pred, f_label
#     # return np.sum(np.abs(y_pred - f_label)) / np.sum(f_label)


# def KL(p, q):
#     if type(p) != torch.Tensor:
#         p = torch.Tensor(p)
#     if type(q) != torch.Tensor:
#         q = torch.Tensor(q)
#     p = p / p.sum()
#     q = q / q.sum()
#     return (p * (p / q).log()).sum()


# def f_aux(y, f, minmax):
#     indices = np.where((f >= minmax[0]) & (f <= minmax[1]))
#     if len(indices[0]) > 0:
#         return len(indices[0]), np.average(np.abs(y[indices] - f[indices])), np.std(np.abs(y[indices] - f[indices]))
#         # return len(indices[0]), y[indices], f[indices]
#     else:
#         return None
#     # return len(indices[0]), np.average(np.abs(y[indices] - f[indices])), np.std(np.abs(y[indices] - f[indices]))


# def f(predictions, minmax=[0.1, 0.2]):
#     res = []
#     for y, f in predictions:
#         r = f_aux(y, f, minmax)
#         print(r)
#         if r is not None:
#             res.append(r)
#     return res


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
