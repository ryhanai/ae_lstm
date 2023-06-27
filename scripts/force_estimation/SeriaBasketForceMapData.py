import os
import tarfile
import numpy as np
import urllib.request
from urllib.error import URLError
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd
import cv2
from KonbiniForceMapData import KonbiniRandomScene


def curate_dataset(num_samples=1000, views=range(2)):
    input_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling3/')
    output_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/basket-filling3-c-1k/')
    all_ids = range(num_samples)
    train_ids, validation_ids, test_ids = np.split(all_ids, [int(len(all_ids)*0.75), int(len(all_ids)*0.875)])

    height = 360
    width = 512
    def f(ids, output_dir, data_type):
        fmaps = []
        rgbs = []
        for j in views:
            rgbs.append([])
        for i in ids:
            print(i)
            fmap = pd.read_pickle(os.path.join(input_dir, f'force_zip{i:05}.pkl'))
            fmap = fmap[:,:,:30]
            fmaps.append(fmap.astype('float32'))
            for j in views:
                rgb = cv2.cvtColor(cv2.imread(os.path.join(input_dir, f'rgb{i:05}_{j:05}.jpg')), cv2.COLOR_BGR2RGB)
                rgb_cropped = rgb[int((720-height)/2):int((720-height)/2+height), int((1280-width)/2):int((1280-width)/2+width)]
                rgbs[j].append(rgb_cropped)
        fmaps = np.array(fmaps)
        for j in views:
            rgbs[j] = np.array(rgbs[j])
        pd.to_pickle(fmaps, os.path.join(output_dir, data_type, f'force_bz2.pkl'), compression='bz2')
        pd.to_pickle(rgbs, os.path.join(output_dir, data_type, f'rgb_bz2.pkl'), compression='bz2')
        if data_type == 'train':
            np.save(os.path.join(output_dir, f'force_bounds.npy'), np.array([np.min(fmaps), np.max(fmaps)]))

    f(train_ids, output_dir, 'train')
    f(validation_ids, output_dir, 'validation')
    f(test_ids, output_dir, 'test')


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
                  root_dir=os.path.join( os.path.expanduser('~'), 'Dataset/dataset2/'),
                  ):

        super().__init__(data_type, minmax, stdev, img_format, root_dir, 'basket-filling3-c-1k')

    def _load_data(self):
        images = pd.read_pickle( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, self.data_type, 'rgb_bz2.pkl'), compression='bz2' )
        forces = pd.read_pickle( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, self.data_type, 'force_bz2.pkl'), compression='bz2' )
        self.force_bounds = np.load( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, 'force_bounds.npy') )

        if self.img_format == 'CWH':
            images = [i.transpose(0,3,1,2) for i in images]
        self.images_raw = images

        self.images = [self._normalization(i.astype(np.float32), (0.0, 255.0)) for i in self.images_raw]
        self.images = torch.from_numpy(np.array(self.images)).float()

        if self.img_format == 'CWH':
            self.forces_raw = forces.transpose(0,3,1,2)
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
    def __init__( self,
                  data_type,
                  minmax=[0.1, 0.9],
                  stdev=0.02,
                  img_format='CWH',
                  root_dir=os.path.join( os.path.expanduser('~'), 'Dataset/dataset2/'),
                  datasize='1k',
                  ):
        SeriaBasketRandomScene.__init__(self,
                  data_type=data_type,
                  minmax=minmax,
                  stdev=stdev,
                  img_format=img_format,
                  root_dir=root_dir,
                  datasize=datasize)

    def __len__(self):
        return len(self.forces)

    def __getitem__(self, idx):
        n_views = self.images.shape[0]
        x_img = self.images[np.random.randint(n_views), idx]
        y_force = self.forces[idx]
        return x_img, y_force


if __name__ == '__main__':
    import time
    import matplotlib.pylab as plt
    
    # load data
    # data_loader = KonbiniRandomSceneDataset('train', minmax=[0.1, 0.9], img_format='CWH')
    
    # x_img, y_force = data_loader[1]

    # images = np.concatenate( (x_data[0], y_data[0]), axis=3 )
    # images = images.transpose(0,2,3,1)

    # # tensor to numpy
    # images = deprocess_img( images, 0.1, 0.9 )

    # # plot joint angles
    # plt.plot(x_data[1], linestyle='dashed', c='k')
    # plt.plot(y_data[1])
    # plt.show()
    
    # # plt images
    # for i in range(images.shape[0]):
    #     cv2.imshow('Video', images[i,:,:,::-1])

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     time.sleep(0.05)
    
    # cv2.destroyAllWindows()
