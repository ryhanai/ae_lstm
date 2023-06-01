import os
import tarfile
import numpy as np
import urllib.request
from urllib.error import URLError
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from eipl.utils import normalization

import pandas as pd
import cv2

def curate_dataset(num_samples=100):
    input_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/konbini-stacked/')
    output_dir = os.path.join(os.path.expanduser('~'), 'Dataset/dataset2/konbini-stacked-c/')
    all_ids = range(num_samples)
    train_ids, validation_ids, test_ids = np.split(all_ids, [int(len(all_ids)*0.75), int(len(all_ids)*0.875)])

    def f(ids, output_dir):
        fmaps = []
        rgbs = []
        for i in ids:
            print(i)
            fmap = pd.read_pickle(os.path.join(input_dir, f'force_zip{i:05}.pkl'), compression='zip')
            fmaps.append(fmap.astype('float32'))
            rgb = cv2.cvtColor(cv2.imread(os.path.join(input_dir, f'rgb{i:05}.jpg')), cv2.COLOR_BGR2RGB)
            rgbs.append(cv2.resize(rgb, (640, 320)))
        pd.to_pickle(np.array(fmaps), os.path.join(output_dir, f'force_bz2.pkl'), compression='bz2')
        pd.to_pickle(np.array(rgbs), os.path.join(output_dir, f'rgb_bz2.pkl'), compression='bz2')

    f(train_ids, os.path.join(output_dir, 'train'))
    f(validation_ids, os.path.join(output_dir, 'validation'))
    f(test_ids, os.path.join(output_dir, 'test'))


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
                  ):
        
        self.data_type  = data_type
        self.minmax     = minmax
        self.stdev      = stdev
        self.img_format = img_format
        self.task_name  = 'konbini-stacked-c'
        self.root_dir   = root_dir
        self.mirror_url = ''

        self.images_raw, self.forces_raw = self._load_data()

        # normalization
        # images: (data_num, c, h, w)
        # forces: (data_num, c, h, w)
        _images = self._normalization(self.images_raw.astype(np.float32), (0.0, 255.0) )
        self.images = torch.from_numpy(_images).float()
        # forcemap is saved in log-scale in the original dataset, so no need to normalize
        self.forces = self.forces_raw

    def _load_data(self):
        images = pd.read_pickle( os.path.join(self.root_dir, self.task_name, self.data_type, 'rgb_bz2.pkl'), compression='bz2' )
        forces = pd.read_pickle( os.path.join(self.root_dir, self.task_name, self.data_type, 'force_bz2.pkl'), compression='bz2' )
        
        if self.img_format == 'CWH':
            images = images.transpose(0,3,1,2)
            forces = forces.transpose(0,3,1,2)

        return images, forces

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def get_data(self, device=None):
        return self.images.to(device), self.forces.to(device)

    def get_raw_data(self):
        return self.images_raw, self.forces_raw


class KonbiniRandomImageDataset(Dataset, KonbiniRandomScene):
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
                  root_dir=os.path.join( os.path.expanduser('~'), '.eipl/'),
                  download=True):
        KonbiniRandomScene.__init__(self,
                  data_type=data_type,
                  minmax=minmax,
                  stdev=stdev,
                  img_format=img_format,
                  root_dir=root_dir,
                  download=download)
        
        self.images_flatten = self.images.reshape( ((-1,) + self.images.shape[2:]) )

        self.transform_affine = transforms.Compose([
            transforms.RandomAffine( degrees=(0, 0), translate=(0.15, 0.15) ),
            transforms.ColorJitter(hue=0.1, saturation=0.1),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
            ])
        self.transform_noise  = transforms.Compose([
            transforms.ColorJitter(contrast=0.5, brightness=0.5),
            ])

    def __len__(self):
        return len(self.images_flatten)

    def __getitem__(self, idx):
        # normalization and convert numpy array to torch tensor
        y_img = self.transform_affine( self.images_flatten[idx] )

        # apply base image transformations to image                                                                                     
        x_img = self.transform_noise( y_img ) + torch.normal(mean=0, std=self.stdev, size=y_img.shape)
        
        return [ x_img, y_img ]


if __name__ == '__main__':
    import cv2
    import time
    import matplotlib.pylab as plt
    from eipl.utils import deprocess_img, tensor2numpy
    
    # load data
    data_loader = KonbiniRandomScene('train', minmax=[0.1, 0.9], img_format='CWH')
    # x_data, y_data = data_loader[1]
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
