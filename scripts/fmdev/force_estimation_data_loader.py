# -*- coding: utf-8 -*-

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import abc
from core.utils import swap


class ForceEstimationDataLoaderBase(object, metaclass=abc.ABCMeta):
    def __init__(self, 
                 image_height, 
                 image_width, 
                 synthetic_data_path, 
                 real_data_path, 
                 crop=128):
        self._image_height = image_height
        self._image_width = image_width
        self._dataset_path = synthetic_data_path
        self._real_dataset_path = real_data_path

    def load_rgb(self, data_id, crop=128, resize=True):
        rgb_path = self.rgb_path_from_id(data_id)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgb = rgb[:, crop:-crop]  # crop the right & left side
        rgb = cv2.resize(rgb, (self._image_width, self._image_height))
        return rgb

    def load_depth(self, data_id, crop=128, resize=True):
        depth_path = self.depth_path_from_id(data_id)
        depth = pd.read_pickle(depth_path, compression='zip')
        depth = depth[:, crop:-crop]
        depth = cv2.resize(depth, (self._image_width, self._image_height))
        depth = np.expand_dims(depth, axis=-1)
        return depth

    def load_segmentation_mask(self, data_id, crop=128, resize=True):
        seg_path = self.segmentation_mask_path_from_id(data_id)
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)    
        seg = seg[:, crop:-crop]
        seg = seg[::2, ::2]  # resize
        return seg

    def load_force(self, data_id, num_z_channels=20, resize=True):
        force_path = self.force_path_from_id(data_id)
        # force = pd.read_pickle(force_path, compression='zip')
        force = pd.read_pickle(force_path, compression='zip')  # compression='infer' didn't work
        force = force[:, :, :num_z_channels]
        return force

    def load_bin_state(self, data_id):
        bin_state_path = self.bin_state_path_from_id(data_id)
        bin_state = pd.read_pickle(bin_state_path)
        return bin_state

    @abc.abstractmethod
    def rgb_path_from_id(self):
        pass

    @abc.abstractmethod
    def depth_path_from_id(self):
        pass

    @abc.abstractmethod
    def segmentation_mask_path_from_id(self):
        pass

    @abc.abstractmethod
    def force_path_from_id(self):
        pass

    @abc.abstractmethod
    def bin_state_path_from_id(self):
        pass

    def load_data(self, ids, data_type, num_z_channels=20):
        total_frames = len(ids)

        if data_type == 'rgb':
            X_rgb = np.empty((total_frames, self._image_height, self._image_width, 3))
            for i in range(0, total_frames):
                print('\rloading RGB ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                X_rgb[i] = self.load_rgb(ids[i])
            X_rgb /= 255.
            return X_rgb

        if data_type == 'depth':
            Y_depth = np.empty((total_frames, self._image_height, self._image_width, 1))
            for i in range(0, total_frames):
                print('\rloading depth ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                Y_depth[i] = self.load_depth(ids[i])
            dmax = np.max(Y_depth)
            dmin = np.min(Y_depth)
            Y_depth = (Y_depth - dmin) / (dmax - dmin)
            return Y_depth

        if data_type == 'segmentation-mask':
            Y_seg = np.empty((total_frames, self._image_height, self._image_width))
            for i in range(0, total_frames):
                print('\rloading segmentation mask ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                Y_seg[i] = self.load_segmentation_mask(ids[i])
            return Y_seg

        if data_type == 'force':
            f = self.load_force(ids[0], num_z_channels=num_z_channels)
            Y_force = np.empty((total_frames,) + f.shape)
            for i in range(0, total_frames):
                print('\rloading force ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                Y_force[i] = self.load_force(ids[i], num_z_channels=num_z_channels)
            fmax = np.max(Y_force)
            Y_force /= fmax
            return Y_force

        if data_type == 'bin-state':
            bin_states = []
            for i in range(0, total_frames):
                print('\rloading bin_state ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                bin_states.append(self.load_bin_state(ids[i]))
            return bin_states

    # def show1(self, scene_data):
    #     rgb, depth, seg, force, bin_state = scene_data
    #     pprint.pprint(bin_state)
    #     plt.figure()
    #     plt.imshow(rgb)
    #     plt.figure()
    #     plt.imshow(depth, cmap='gray')
    #     plt.figure()
    #     plt.imshow(seg)
    #     visualize_forcemaps(force)
    #     plt.show()

    def load_data_for_rgb2fmap(self, train_mode=False, test_mode=False, load_bin_state=False, num_z_channels=20):
        """
        Returns:
            train_data, valid_data, test_data
            train_data := (train_rgb, train_force)
        """
        assert train_mode ^ test_mode, 'either train_mode or test_mode must be True'

        if train_mode:
            train_rgb = self.load_data(self._train_ids, 'rgb')
            valid_rgb = self.load_data(self._valid_ids, 'rgb')
            train_force = self.load_data(self._train_ids, 'force', num_z_channels=num_z_channels)
            valid_force = self.load_data(self._valid_ids, 'force', num_z_channels=num_z_channels)
            return (train_rgb, train_force), (valid_rgb, valid_force)

        if test_mode:
            test_rgb = self.load_data(self._test_ids, 'rgb')
            test_force = self.load_data(self._test_ids, 'force', num_z_channels=num_z_channels)
            if load_bin_state:
                test_bin_state = self.load_data(self._test_ids, 'bin-state')
                return test_rgb, test_force, test_bin_state
            else:
                return test_rgb, test_force

    def load_data_for_dseg2fmap(self, train_mode=True, test_mode=False, load_bin_state=False):
        """
        Returns:
            train_data, valid_data, test_data
            train_data := (train_depth, train_seg), train_force
        """
        assert train_mode ^ test_mode, 'either train_mode or test_mode must be True'

        if train_mode:
            train_depth = self.load_data(self._train_ids, 'depth')
            valid_depth = self.load_data(self._valid_ids, 'depth')
            train_seg = self.load_data(self._train_ids, 'segmentation-mask')
            valid_seg = self.load_data(self._valid_ids, 'segmentation-mask')
            train_force = self.load_data(self._train_ids, 'force')
            valid_force = self.load_data(self._valid_ids, 'force')
            return ((train_depth, train_seg), train_force), ((valid_depth, valid_seg), valid_force)

        if test_mode:
            test_depth = self.load_data(self._test_ids, 'depth')
            test_seg = self.load_data(self._test_ids, 'segmentation-mask')
            test_force = self.load_ddta(self._test_ids, 'force')
            return (test_depth, test_seg), test_force

    def load_real_data(self, ids, image_center_offset=np.array([-40, 0]), scale=0.6):
        total_frames = len(ids)
        X_rgb = np.empty((total_frames, self._image_height, self._image_width, 3))
        resized_shape = (int(scale * 760), int(scale * 1280))
        target_size = np.array([self._image_height, self._image_width])
        crop = ((resized_shape - target_size) / 2).astype(np.int32)

        for i, id in enumerate(ids):
            print('\rloading RGB ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            filepath = os.path.join(self._real_dataset_path, '{:05d}.png'.format(id))
            rgb = plt.imread(filepath)
            resized_rgb = cv2.resize(rgb, swap(resized_shape), cv2.INTER_AREA)
            top_left = crop + image_center_offset
            bottom_right = crop + image_center_offset + target_size
            X_rgb[i] = resized_rgb[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        return X_rgb

    def load_real_data_for_rgb2fmap(self, train_mode=False, test_mode=False):
        """
        Returns:
            train_data, valid_data, test_data
        """
        assert train_mode ^ test_mode, 'either train_mode or test_mode must be True'

        if train_mode:
            train_rgb = self.load_real_data(self._train_real_ids)
            valid_rgb = self.load_real_data(self._valid_real_ids)
            return train_rgb, valid_rgb

        if test_mode:
            test_rgb = self.load_real_data(self._test_real_ids)
            return test_rgb


class ForceEstimationDataLoader(ForceEstimationDataLoaderBase):
    def __init__(self,
                 synthetic_data_path, real_data_path,
                 image_height=360, image_width=512,
                 start_seq=1, n_seqs=1500,
                 start_frame=3, n_frames=3,
                 real_start_frame=1, real_n_frames=294,
                 ):

        super().__init__(image_height, image_width, synthetic_data_path, real_data_path)

        # configuration for synthetic data    
        # self.__start_seq = start_seq
        # self._n_seqs = n_seqs
        # self._start_frame = start_frame
        # self._n_frames = n_frames

        # configuration for real data
        self._real_start_frame = real_start_frame
        self._real_n_frames = real_n_frames

        x, y = np.mgrid[start_seq:start_seq + n_seqs, start_frame:start_frame + n_frames]
        data_ids = list(zip(x.ravel(), y.ravel()))
        self._train_ids, self._valid_ids, self._test_ids = np.split(data_ids, [int(len(data_ids)*.75), int(len(data_ids)*.875)])

        np.random.seed(0)
        self._train_real_ids, self._valid_real_ids, self._test_real_ids = np.split(np.random.permutation(range(self._real_start_frame, self._real_start_frame+self._real_n_frames)), [int(self._real_n_frames*.75), int(self._real_n_frames*.875)])

    def rgb_path_from_id(self, data_id):
        seqNo, frameNo = data_id
        return os.path.join(self._dataset_path, str(seqNo), 'rgb{:05d}.jpg'.format(frameNo))

    def depth_path_from_id(self, data_id):
        seqNo, frameNo = data_id
        return os.path.join(self._dataset_path, str(seqNo), 'depth_zip{:05d}.pkl'.format(frameNo))

    def segmentation_mask_path_from_id(self, data_id):
        seqNo, frameNo = data_id
        return os.path.join(self._dataset_path, str(seqNo), 'seg{:05d}.png'.format(frameNo))            

    def force_path_from_id(self, data_id):
        seqNo, frameNo = data_id
        return os.path.join(self._dataset_path, str(seqNo), 'force_zip{:05d}.pkl'.format(frameNo))

    def bin_state_path_from_id(self, data_id):
        seqNo, frameNo = data_id
        return os.path.join(self._dataset_path, str(seqNo), 'bin_state{:05d}.pkl'.format(frameNo))


class ForceEstimationDataLoaderNoSeq(ForceEstimationDataLoaderBase):
    def __init__(self,
                 synthetic_data_path, real_data_path,
                 image_height=360, image_width=512,
                 start_frame=0, n_frames=5000,
                 real_start_frame=0, real_n_frames=40,
                 ):

        super().__init__(image_height, image_width, synthetic_data_path, real_data_path)

        self._real_start_frame = real_start_frame
        self._real_n_frames = real_n_frames

        data_ids = range(start_frame, start_frame + n_frames)
        self._train_ids, self._valid_ids, self._test_ids = np.split(data_ids,
                                                                    [int(len(data_ids)*.75), int(len(data_ids)*.875)])

        self._train_real_ids = []
        self._valid_real_ids = []
        self._test_real_ids = range(real_start_frame, real_start_frame + real_n_frames)

    def rgb_path_from_id(self, data_id):
        frameNo = data_id
        return os.path.join(self._dataset_path, 'rgb{:05d}.jpg'.format(frameNo))

    def depth_path_from_id(self, data_id):
        frameNo = data_id
        return os.path.join(self._dataset_path, 'depth_zip{:05d}.pkl'.format(frameNo))

    def segmentation_mask_path_from_id(self, data_id):
        frameNo = data_id
        return os.path.join(self._dataset_path, 'seg{:05d}.png'.format(frameNo))            

    def force_path_from_id(self, data_id):
        frameNo = data_id
        return os.path.join(self._dataset_path, 'force_zip{:05d}.pkl'.format(frameNo))

    def bin_state_path_from_id(self, data_id):
        frameNo = data_id
        return os.path.join(self._dataset_path, 'bin_state{:05d}.pkl'.format(frameNo))
