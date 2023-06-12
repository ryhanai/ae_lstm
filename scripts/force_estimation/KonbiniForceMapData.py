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
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data


def curate_dataset(num_samples=100):
    input_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/konbini-stacked/')
    output_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/konbini-stacked-c/')
    all_ids = range(num_samples)
    train_ids, validation_ids, test_ids = np.split(all_ids, [int(len(all_ids)*0.75), int(len(all_ids)*0.875)])

    height = 14 * 24
    width = height * 2
    def f(ids, output_dir, data_type):
        fmaps = []
        rgbs = []
        for i in ids:
            print(i)
            fmap = pd.read_pickle(os.path.join(input_dir, f'force_zip{i:05}.pkl'), compression='zip')
            fmap = fmap[:,:,:30]
            fmaps.append(fmap.astype('float32'))
            rgb = cv2.cvtColor(cv2.imread(os.path.join(input_dir, f'rgb{i:05}.jpg')), cv2.COLOR_BGR2RGB)
            rgbs.append(cv2.resize(rgb, (width, height)))
        fmaps = np.array(fmaps)
        rgbs = np.array(rgbs)
        pd.to_pickle(fmaps, os.path.join(output_dir, data_type, f'force_bz2.pkl'), compression='bz2')
        pd.to_pickle(rgbs, os.path.join(output_dir, data_type, f'rgb_bz2.pkl'), compression='bz2')
        if data_type == 'train':
            np.save(os.path.join(output_dir, f'force_bounds.npy'), np.array([np.min(fmaps), np.max(fmaps)]))

    f(train_ids, output_dir, 'train')
    f(validation_ids, output_dir, 'validation')
    f(test_ids, output_dir, 'test')


class KonbiniRandomScene:
    """KonbiniRandomScene dataset.

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
        
        self.data_type  = data_type
        self.minmax     = minmax
        self.stdev      = stdev
        self.img_format = img_format
        self.task_name  = 'konbini-stacked-c'
        self.root_dir   = root_dir
        self.datasize   = datasize
        self.mirror_url = ''

        self.images_raw, self.forces_raw, self.force_bounds = self._load_data()

        # normalization
        # images: (data_num, c, h, w)
        # forces: (data_num, c, h, w)
        _images = self._normalization(self.images_raw.astype(np.float32), (0.0, 255.0))
        self.images = torch.from_numpy(_images).float()
        # forcemap is saved in log-scale in the original dataset
        _forces = self._normalization(self.forces_raw, self.force_bounds)
        self.forces = torch.from_numpy(_forces).float()

    def _load_data(self):
        images = pd.read_pickle( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, self.data_type, 'rgb_bz2.pkl'), compression='bz2' )
        forces = pd.read_pickle( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, self.data_type, 'force_bz2.pkl'), compression='bz2' )
        force_bounds = np.load( os.path.join(self.root_dir, self.task_name+'-'+self.datasize, 'force_bounds.npy') )
        
        if self.img_format == 'CWH':
            images = images.transpose(0,3,1,2)
            forces = forces.transpose(0,3,1,2)

        return images, forces, force_bounds

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def get_data(self, device=None):
        return self.images.to(device), self.forces.to(device)

    def get_raw_data(self):
        return self.images_raw, self.forces_raw


class KonbiniRandomSceneDataset(Dataset, KonbiniRandomScene):
    """AIREC_sample dataset.

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
        KonbiniRandomScene.__init__(self,
                  data_type=data_type,
                  minmax=minmax,
                  stdev=stdev,
                  img_format=img_format,
                  root_dir=root_dir,
                  datasize=datasize)

        # self.images_flatten = self.images.reshape( ((-1,) + self.images.shape[2:]) )

        # self.transform_affine = transforms.Compose([
        #     transforms.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03)),
        #     transforms.ColorJitter(hue=0.1, saturation=0.1),
        #     transforms.RandomAutocontrast(),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip()
        #     ])
        # self.transform_noise  = transforms.Compose([
        #     transforms.ColorJitter(contrast=0.1, brightness=0.1),
        #     ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # normalization and convert numpy array to torch tensor
        # y_img = self.transform_affine(self.images[idx])
        # apply base image transformations to image                                                                                     
        # x_img = self.transform_noise(y_img) + torch.normal(mean=0, std=self.stdev, size=y_img.shape)

        x_img = self.images[idx]
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
