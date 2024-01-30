import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from eipl_print_func import *
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
        img_format="CWH",
        root_dir=Path(os.path.expanduser("~")) / "Dataset/forcemap/",
        task_name="tabletop240125",
        num_samples=1000,
        num_views=3,
        view_index=None,
        method="geometry-aware",
    ):
        self.data_type = data_type
        self.minmax = minmax
        self.img_format = img_format
        self.task_name = task_name
        self.root_dir = root_dir
        self.mirror_url = ""

        self._input_dir = root_dir / task_name

        self._num_samples = 1000
        self._num_views = 3
        self._load_data()
        self._view_index = view_index

        self._force_bounds = self._compute_force_bounds()

        methods = {
            "isotropic": 0,
            "geometry-aware": 1,
            "sdf": 2,
        }
        self._method = methods[method]

    # def get_data(self, device=None):
    #     return self.images.to(device), self.forces.to(device)

    def _load_data(self):
        self._all_ids = range(self._num_samples)
        self._train_ids, self._validation_ids, self._test_ids = np.split(
            self._all_ids, [int(len(self._all_ids) * 0.75), int(len(self._all_ids) * 0.875)]
        )
        if self.data_type == "train":
            self._ids = self._train_ids
        elif self.data_type == "validation":
            self._ids = self._validation_ids
        elif self.data_type == "test":
            self._ids = self._test_ids

    def _compute_force_bounds(self):
        return np.array([1e-5, 1e-0])  # currently,, we use a fixed force range

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def get_raw_data(self):
        return self.images_raw, self.forces_raw

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        imgs = []
        fmaps = []
        for i in idx:
            imgs.append(self.load_image(i))
            fmaps.append(self.load_fmap(i))
        fmaps = np.array(fmaps)
        y_force = torch.from_numpy(fmaps).float()
        imgs = np.array(imgs)
        x_img = torch.from_numpy(imgs).float()
        return x_img, y_force

    def load_fmap(self, idx):
        scene_idx = self._ids[idx]
        fmap = pd.read_pickle(self._input_dir / f"force_zip{scene_idx:05}.pkl")[self._method]
        fmap = fmap[:, :, :30].astype("float32")
        fmap = np.clip(fmap, self._force_bounds[0], self._force_bounds[1])
        fmap = np.log(fmap)  # force_raw (in log scale)
        fmap = fmap.transpose(2, 0, 1)
        fmap = self._normalization(fmap, np.log(self._force_bounds))
        return fmap

    def load_image(self, idx):
        scene_idx = self._ids[idx]
        n_views = self._num_views
        if self._view_index == None:
            view_idx = np.random.randint(n_views)
        else:
            view_idx = self._view_index

        rgb = cv2.cvtColor(cv2.imread(str(self._input_dir / f"rgb{idx:05}_{view_idx:05}.jpg")), cv2.COLOR_BGR2RGB)
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

    def get_specific_view_and_force(self, idx, view_idx):
        n_views = self.images.shape[0]
        assert view_idx < n_views, f"the dataset has {n_views} views, but view_idx=={view_idx} was specified"
        x_img = self.images[view_idx, idx]
        y_force = self.forces[idx]
        return x_img, y_force
