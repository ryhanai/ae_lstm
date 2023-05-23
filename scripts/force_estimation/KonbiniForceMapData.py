import os
import tarfile
import numpy as np
import urllib.request
from urllib.error import URLError
import torch
from torchvision import transforms
from torch.utils.data import Dataset
# from eipl.utils import normalization


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
                  root_dir=os.path.join( os.path.expanduser('~'), 'Dataset/dataset2/konbini-stacked/'),        
                  download=False):
        
        self.data_type  = data_type
        self.minmax     = minmax
        self.stdev      = stdev
        self.img_format = img_format
        self.task_name  = 'force_prediction'
        self.root_dir   = root_dir
        self.tar_data_path = os.path.join(self.root_dir, self.task_name+'.tar' )
        self.mirror_url = ''

        if download:
            self._download()

        # load npy data
        self.joint_bounds = self._load_bounds()
        self.images_raw, self.joints_raw = self._load_data()

        # normalization
        # images: (data_num, seq_len, c, h, w)
        # joints: (data_num, seq_len, 8)
        _images = self._normalization(self.images_raw.astype(np.float32), (0.0, 255.0) )
        _joints = self._normalization(self.joints_raw.astype(np.float32), self.joint_bounds )
        self.images = torch.from_numpy(_images).float()
        self.joints = torch.from_numpy(_joints).float()

    def _check_exists(self):
        return os.path.isfile( self.tar_data_path )

    def _download(self):
        """Download the data if it doesn't exist already."""
        os.makedirs(self.root_dir, exist_ok=True)

        # download files
        try:
            if not self._check_exists():
                print(f"Downloading {self.mirror_url}")
                urllib.request.urlretrieve(self.mirror_url, self.tar_data_path)

            with tarfile.open(self.tar_data_path, 'r:tar') as tar:
                tar.extractall(path=self.root_dir)

        except URLError as error:
            raise RuntimeError(f"Error downloading")

    def _load_bounds(self):
        joint_bounds = np.load( os.path.join(self.root_dir, self.task_name, 'joint_bounds.npy') )
        return joint_bounds

    def _load_data(self):
        joints = np.load( os.path.join(self.root_dir, self.task_name, self.data_type, 'joints.npy') )
        images = np.load( os.path.join(self.root_dir, self.task_name, self.data_type, 'images.npy') )
        
        if self.img_format == 'CWH':
            images = images.transpose(0,1,4,2,3)

        return images, joints

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def get_data(self, device=None):
        return self.images.to(device), self.joints.to(device)

    def get_raw_data(self):
        return self.images_raw, self.joints_raw


class GraspBottleDataset(Dataset, GraspBottle):
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
        GraspBottle.__init__(self,
                  data_type=data_type,
                  minmax=minmax,
                  stdev=stdev,
                  img_format=img_format,
                  root_dir=root_dir,
                  download=download)
        
        self.transform = transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.1)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # normalization and convert numpy array to torch tensor
        y_img   = self.images[idx]
        y_joint = self.joints[idx]

        # apply base image transformations to image
        x_img = self.transform( self.images[idx] )
        x_img = x_img + torch.normal(mean=0, std=self.stdev, size=x_img.shape)

        # apply gaussian noise to joint angles
        x_joint = self.joints[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)

        return [ [x_img, x_joint], [y_img, y_joint] ]


class GraspBottleImageDataset(GraspBottleDataset):
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
        GraspBottleDataset.__init__(self,
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
    data_loader = GraspBottleDataset('train', minmax=[0.1, 0.9], img_format='CWH')
    x_data, y_data = data_loader[1]
    images = np.concatenate( (x_data[0], y_data[0]), axis=3 )
    images = images.transpose(0,2,3,1)

    # tensor to numpy
    images = deprocess_img( images, 0.1, 0.9 )

    # plot joint angles
    plt.plot(x_data[1], linestyle='dashed', c='k')
    plt.plot(y_data[1])
    plt.show()
    
    # plt images
    for i in range(images.shape[0]):
        cv2.imshow('Video', images[i,:,:,::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
    
    cv2.destroyAllWindows()
