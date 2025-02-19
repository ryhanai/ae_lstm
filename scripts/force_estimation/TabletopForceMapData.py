import json
import os
import re
import yaml
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from force_estimation.eipl_print_func import *
from torch.utils.data import Dataset


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. ind    ataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indat    aRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


class TabletopRandomSceneDataset(Dataset):
    """TabletopRandomScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root_dir (string, optional):   Root directory of the data set.
        method (string, optional):  "isotropic" | "geometry-aware" | "sdf"
    """

    def __init__(
        self,
        data_type,
        minmax=[0.1, 0.9],
        fminmax=[1e-5, 1e-0],
        img_format="CWH",
        root_dir=Path(os.path.expanduser("~")) / "Dataset/forcemap/",
        task_name="tabletop240125",
        num_samples=1000,
        num_views=3,
        method="geometry-aware",
    ):
        self.data_type = data_type
        self.minmax = minmax
        self.fminmax = np.array(fminmax)
        self.img_format = img_format
        self.task_name = task_name
        self.root_dir = root_dir
        self.mirror_url = ""
        
        self._num_samples = num_samples
        self._num_views = num_views
        self._load_data()

        self._force_bounds = self._compute_force_bounds()

        print_info(f"label = {method}")
        methods = {
            "isotropic": 0,
            "geometry-aware": 1,
            "sdf": 2,
        }
        self._method = methods[method]

    # def get_data(self, device=None):
    #     return self.images.to(device), self.forces.to(device)

    def _scan_ids(self, dataset_name):
        p = str(Path(self.root_dir) / dataset_name / 'bin_state*.pkl')
        bin_files = glob.glob(str(p))
        ids = [re.findall('bin_state(.*).pkl', x)[0] for x in bin_files]
        return [(dataset_name, i) for i in ids]

    def _load_data(self, split=[0.75, 0.875]):
        sub_dataset_def_file = Path(self.root_dir) / self.task_name / 'sub_datasets.yaml'
        if os.path.isfile(sub_dataset_def_file):
            with open(sub_dataset_def_file) as f:
                sub_datasets = yaml.safe_load(f)['sub_datasets']
        else:
            sub_datasets = [self.task_name]

        print(f'LOAD: {sub_datasets}')
        self._all_ids = []
        for sub_dataset in sub_datasets:
            self._all_ids.extend(self._scan_ids(sub_dataset))

        # shuffle ids
        np.random.seed(42)
        self._all_ids = np.random.permutation(self._all_ids)

        train_ids, validation_ids, test_ids = np.split(
            self._all_ids, [int(len(self._all_ids) * split[0]), int(len(self._all_ids) * split[1])]
        )
        if self.data_type == "train":
            self._ids = train_ids
        elif self.data_type == "validation":
            self._ids = validation_ids
        elif self.data_type == "test":
            self._ids = test_ids

        self._ids = [(x[0], int(x[1])) for x in self._ids]

    def _compute_force_bounds(self):
        return self.fminmax

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def get_raw_data(self):
        return self.images_raw, self.forces_raw

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx, view_idx=None):
        if view_idx == None:
            view_idx = np.random.randint(self._num_views)
        assert (
            view_idx < self._num_views
        ), f"the dataset has {self._num_views} views, but view_idx=={view_idx} was specified"

        imgs = []
        fmaps = []
        for i in idx:
            imgs.append(self.load_image(i, view_idx))
            fmaps.append(self.load_fmap(i))
        fmaps = np.array(fmaps)
        y_force = torch.from_numpy(fmaps).float()
        imgs = np.array(imgs)
        x_img = torch.from_numpy(imgs).float()
        return x_img, y_force

    def load_fmap(self, idx):
        dataset_name, scene_idx = self._ids[idx]
        fmap = pd.read_pickle(self.root_dir / dataset_name / f"force_zip{scene_idx:05}.pkl")[self._method]
        fmap = fmap[:, :, :30].astype("float32")

        if self._method == 0 or self._method == 1:
            fmap = np.clip(fmap, self._force_bounds[0], self._force_bounds[1])
            fmap = np.log(fmap)  # force_raw (in log scale)
            fmap = fmap.transpose(2, 0, 1)
            fmap = self._normalization(fmap, np.log(self._force_bounds))
        else:
            dist_bounds = [-0.001, 0.02]
            fmap = np.clip(-fmap, dist_bounds[0], dist_bounds[1])
            fmap = fmap.transpose(2, 0, 1)
            fmap = self._normalization(fmap, dist_bounds)
        return fmap

    def load_image(self, idx, view_idx):
        dataset_name, scene_idx = self._ids[idx]
        rgb = cv2.cvtColor(cv2.imread(str(self.root_dir / dataset_name / f"rgb{scene_idx:05}_{view_idx:05}.jpg")), cv2.COLOR_BGR2RGB)
        if self.img_format == "CWH":
            rgb = rgb.transpose(2, 0, 1)
        rbg = rgb.astype("float32")
        rgb = self._normalization(rgb, (0.0, 255.0))
        return rgb

    def reformat(self, d):
        # used for visualization with fmap_visualizer
        d = np.array(d)
        d = d.transpose(1, 2, 0)
        e = np.zeros((80, 80, 40))
        e[:, :, :30] = d
        return e

    def load_bin_state(self, idx):
        dataset_name, scene_idx = self._ids[idx]
        p = Path(self.root_dir) / dataset_name / f"bin_state{scene_idx:05d}.pkl"
        return pd.read_pickle(p)

    # def get_specific_view_and_force(self, idx, view_idx):
    #     assert (
    #         view_idx < self._num_views
    #     ), f"the dataset has {self._num_views} views, but view_idx=={view_idx} was specified"

    #     x_img = self.images[view_idx, idx]
    #     y_force = self.forces[idx]
    #     return x_img, y_force
