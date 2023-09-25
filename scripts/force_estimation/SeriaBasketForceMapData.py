import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd
import cv2
import json
from eipl_print_func import *


from KonbiniForceMapData import KonbiniRandomScene


def curate_dataset(num_samples=2000,
                   views=range(3),
                   ):
    # input_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling3/')
    # output_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling3-c-1k/')
    input_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling230918/')
    output_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling230918-c-2k/')
    all_ids = range(num_samples)
    train_ids, validation_ids, test_ids = np.split(all_ids, [int(len(all_ids)*0.75), int(len(all_ids)*0.875)])

    height = 360
    width = 512

    def f(ids, output_dir, data_type, fmin=1e-8, fmax=1e-3):
        os.mkdir(os.path.join(output_dir, data_type))
        fmaps = []
        rgbs = []
        bss = []
        bounds = np.log([fmin, fmax])

        for j in views:
            rgbs.append([])
        for i in ids:
            print(i)
            fmap = pd.read_pickle(os.path.join(input_dir, f'force_zip{i:05}.pkl'))
            fmap = fmap[:, :, :30].astype('float32')
            fmap = np.clip(fmap, fmin, fmax)
            fmap = np.log(fmap)
            fmaps.append(fmap)

            bss.append(pd.read_pickle(os.path.join(input_dir, f'bin_state{i:05}.pkl')))
            for j in views:
                rgb = cv2.cvtColor(cv2.imread(os.path.join(input_dir, f'rgb{i:05}_{j:05}.jpg')), cv2.COLOR_BGR2RGB)
                rgb_cropped = rgb[int((720-height)/2):int((720-height)/2+height), int((1280-width)/2):int((1280-width)/2+width)]
                rgbs[j].append(rgb_cropped)
        fmaps = np.array(fmaps)
        for j in views:
            rgbs[j] = np.array(rgbs[j])
        pd.to_pickle(fmaps, os.path.join(output_dir, data_type, 'force_bz2.pkl'), compression='bz2')
        pd.to_pickle(rgbs, os.path.join(output_dir, data_type, 'rgb_bz2.pkl'), compression='bz2')
        pd.to_pickle(bss, os.path.join(output_dir, data_type, 'bin_state_bz2.pkl'), compression='bz2')
        if data_type == 'train':
            np.save(os.path.join(output_dir, 'force_bounds.npy'), bounds)
            # np.save(os.path.join(output_dir, 'force_bounds.npy'), np.array([np.min(fmaps), np.max(fmaps)]))

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print_error(f'Direcotry {output_dir} already exists.')
        return

    f(train_ids, output_dir, 'train')
    f(validation_ids, output_dir, 'validation')
    f(test_ids, output_dir, 'test')

    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        dataset_descriptor = {
            "data loader": "SeriaBasketRandomSceneDataset",
            "forcemap": "seria_basket"
            }
        json.dump(dataset_descriptor, f, indent=4)


class SeriaBasketRandomScene(KonbiniRandomScene):
    """SeriaBasketRandomScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root (string, optional):   Root directory of the data set, default is saved in the '~/epil/'.
        download (bool, optional): If True, downloads the dataset from the internet and
                                   puts it in root directory. If dataset is already downloaded, it is not
                                   downloaded again.
    """
    def __init__( self,
                  data_type,
                  minmax=[0.1, 0.9],
                  stdev=0.02,
                  img_format='CWH',
                  root_dir=os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/'),
                  ):

        super().__init__(data_type, minmax, stdev, img_format, root_dir, 'basket-filling230918-c-2k')

    def _load_data(self):
        images = pd.read_pickle(os.path.join(self.root_dir, self.task_name, self.data_type, 'rgb_bz2.pkl'), compression='bz2')
        forces = pd.read_pickle(os.path.join(self.root_dir, self.task_name, self.data_type, 'force_bz2.pkl'), compression='bz2')
        self.force_bounds = np.load(os.path.join(self.root_dir, self.task_name, 'force_bounds.npy'))

        if self.img_format == 'CWH':
            images = [i.transpose(0, 3, 1, 2) for i in images]
        self.images_raw = images

        self.images = [self._normalization(i.astype(np.float32), (0.0, 255.0)) for i in self.images_raw]
        self.images = torch.from_numpy(np.array(self.images)).float()

        if self.img_format == 'CWH':
            self.forces_raw = forces.transpose(0, 3, 1, 2)
        # forcemap is saved in log-scale in the original dataset
        _forces = self._normalization(self.forces_raw, self.force_bounds)
        self.forces = torch.from_numpy(_forces).float()

    def get_data(self, device=None):
        return self.images.to(device), self.forces.to(device)


class SeriaBasketRandomSceneDataset(Dataset, SeriaBasketRandomScene):
    """SeriaBasketRandomScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root (string, optional):   Root directory of the data set, default is saved in the '~/epil/'.
        download (bool, optional): If True, downloads the dataset from the internet and
                                   puts it in root directory. If dataset is already downloaded, it is not
                                   downloaded again.
    """
    def __init__(self,
                 data_type,
                 minmax=[0.1, 0.9],
                 stdev=0.02,
                 img_format='CWH',
                 root_dir=os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/'),
                 view_index=None,
                 ):
        SeriaBasketRandomScene.__init__(self,
                                        data_type=data_type,
                                        minmax=minmax,
                                        stdev=stdev,
                                        img_format=img_format,
                                        root_dir=root_dir,
                                        )
        self._view_index = view_index

    def __len__(self):
        return len(self.forces)

    def __getitem__(self, idx):
        n_views = self.images.shape[0]
        if self._view_index == None:
            x_img = self.images[np.random.randint(n_views), idx]
        else:
            x_img = self.images[self._view_index, idx]
        y_force = self.forces[idx]
        return x_img, y_force

    def get_specific_view_and_force(self, idx, view_idx):
        n_views = self.images.shape[0]
        assert view_idx < n_views, f'the dataset has {n_views} views, but view_idx=={view_idx} was specified'
        x_img = self.images[view_idx, idx]
        y_force = self.forces[idx]
        return x_img, y_force


class SeriaBasketRealScene(KonbiniRandomScene):
    """SeriaBasketRealScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root (string, optional):   Root directory of the data set, default is saved in the '~/epil/'.
        download (bool, optional): If True, downloads the dataset from the internet and
                                   puts it in root directory. If dataset is already downloaded, it is not
                                   downloaded again.
    """
    def __init__(self,
                 data_type,
                 minmax=[0.1, 0.9],
                 stdev=0.02,
                 img_format='CWH',
                 root_dir=os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/'),
                 ):

        super().__init__(data_type, minmax, stdev, img_format, root_dir, 'basket-filling2-real')

    def _load_data(self):
        height = 360
        width = 512

        images = []
        for i in range(1, 295):
            path = os.path.join(self.root_dir, self.task_name, f'{i:05}.png')
            print(path)
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            rgb_cropped = rgb[int((720-height)/2):int((720-height)/2+height), int((1280-width)/2):int((1280-width)/2+width)]
            images.append(rgb_cropped)

        if self.img_format == 'CWH':
            images = [img.transpose(2, 0, 1) for img in images]
        self.images_raw = images

        self.images = [self._normalization(i.astype(np.float32), (0.0, 255.0)) for i in self.images_raw]
        self.images = torch.from_numpy(np.array(self.images)).float()

    def get_data(self, device=None):
        return self.images.to(device), None


class SeriaBasketRealSceneDataset(Dataset, SeriaBasketRealScene):
    """SeriaBasketRealScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root (string, optional):   Root directory of the data set, default is saved in the '~/epil/'.
        download (bool, optional): If True, downloads the dataset from the internet and
                                   puts it in root directory. If dataset is already downloaded, it is not
                                   downloaded again.
    """
    def __init__(self,
                 data_type,
                 minmax=[0.1, 0.9],
                 stdev=0.02,
                 img_format='CWH',
                 root_dir=os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/'),
                 ):
        SeriaBasketRealScene.__init__(self,
                                      data_type=data_type,
                                      minmax=minmax,
                                      stdev=stdev,
                                      img_format=img_format,
                                      root_dir=root_dir,
                                      )

    def __len__(self):
        return len(self.forces)

    def __getitem__(self, idx):
        x_img = self.images[idx]
        return x_img, None
