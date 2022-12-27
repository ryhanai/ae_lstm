# -*- coding: utf-8 -*-

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ForceEstimationDataLoader:
    def __init__(self,
                 synthetic_data_path, real_data_path,
                 image_height=360, image_width=512,
                 num_classes=62,
                 start_seq=1, n_seqs=1500,
                 start_frame=3, n_frames=3,
                 real_start_frame=1, real_n_frames=294,
                 ):

        self._dataset_path = synthetic_data_path
        self._real_dataset_path = real_data_path
        self._image_height = image_height
        self._image_width = image_width
        self._num_classes = num_classes

        # configuration for synthetic data    
        # self.__start_seq = start_seq
        # self._n_seqs = n_seqs
        # self._start_frame = start_frame
        # self._n_frames = n_frames

        # configuration for real data
        self._real_start_frame = real_start_frame
        self._real_n_frames = real_n_frames

        self._crop = 128

        x, y = np.mgrid[start_seq:start_seq + n_seqs, start_frame:start_frame + n_frames]
        data_ids = list(zip(x.ravel(), y.ravel()))
        self._train_ids, self._valid_ids, self._test_ids = np.split(data_ids, [int(len(data_ids)*.75), int(len(data_ids)*.875)])

        np.random.seed(0)
        self._train_real_ids, self._valid_real_ids, self._test_real_ids = np.split(np.random.permutation(range(self._real_start_frame, self._real_start_frame+self._real_n_frames)), [int(self._real_n_frames*.75), int(self._real_n_frames*.875)])

    def load_rgb(self, data_id, resize=True):
        seqNo, frameNo = data_id
        rgb_path = os.path.join(self._dataset_path, str(seqNo), 'rgb{:05d}.jpg'.format(frameNo))
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgb = rgb[:,self._crop:-self._crop] # crop the right & left side
        rgb = cv2.resize(rgb, (self._image_width, self._image_height))
        return rgb

    def load_depth(self, data_id, resize=True):
        seqNo, frameNo = data_id
        depth_path = os.path.join(self._dataset_path, str(seqNo), 'depth_zip{:05d}.pkl'.format(frameNo))
        depth = pd.read_pickle(depth_path, compression='zip')
        depth = depth[:, self._crop:-self._crop]
        depth = cv2.resize(depth, (self._image_width, self._image_height))
        depth = np.expand_dims(depth, axis=-1)
        return depth

    def load_segmentation_mask(self, data_id, resize=True):
        seqNo, frameNo = data_id
        seg_path = os.path.join(self._dataset_path, str(seqNo), 'seg{:05d}.png'.format(frameNo))            
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)    
        seg = seg[:, self._crop:-self._crop]
        seg = seg[::2, ::2]  # resize
        return seg

    def load_force(self, data_id, resize=True):
        seqNo, frameNo = data_id
        force_path = os.path.join(self._dataset_path, str(seqNo), 'force_zip{:05d}.pkl'.format(frameNo))
        force = pd.read_pickle(force_path, compression='zip')
        force = force[:, :, :20]
        return force

    def load_bin_state(self, data_id):
        seqNo, frameNo = data_id
        bin_state_path = os.path.join(self._dataset_path, str(seqNo), 'bin_state{:05d}.pkl'.format(frameNo))
        bin_state = pd.read_pickle(bin_state_path)
        return bin_state

    def load_data(self, ids, data_type):
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
            Y_force = np.empty((total_frames, 40, 40, 20))
            for i in range(0, total_frames):
                print('\rloading force ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
                Y_force[i] = self.load_force(ids[i])
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

    def load_data_for_rgb2fmap(self, train_mode=False, test_mode=False, load_bin_state=False):
        """
        Returns:
            train_data, valid_data, test_data
            train_data := (train_rgb, train_force)
        """
        assert train_mode ^ test_mode, 'either train_mode or test_mode must be True'

        if train_mode:
            train_rgb = self.load_data(self._train_ids, 'rgb')
            valid_rgb = self.load_data(self._valid_ids, 'rgb')
            train_force = self.load_data(self._train_ids, 'force')
            valid_force = self.load_data(self._valid_ids, 'force')
            return (train_rgb, train_force), (valid_rgb, valid_force)

        if test_mode:
            test_rgb = self.load_data(self._test_ids, 'rgb')
            test_force = self.load_data(self._test_ids, 'force')
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

    def load_real_data(self, ids):
        total_frames = len(ids)
        # c = (-40, 25)
        c = (-30, 80)
        crop = 64
        X_rgb = np.empty((total_frames, self._image_height, self._image_width, 3))

        for i, id in enumerate(ids):
            print('\rloading RGB ... {}({:3.1f}%)'.format(i, i/total_frames*100.), end='')
            filepath = os.path.join(self._real_dataset_path, '{:05d}.png'.format(id))
            img = plt.imread(filepath)
            X_rgb[i] = img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]

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
