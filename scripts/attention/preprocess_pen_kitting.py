# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from core.utils import Dataset


class URDataset(Dataset):

    arm_joint_order = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
    ]

    def __init__(self, name, mode, **kargs):
        '''
        Args:
            mode: 'train' or 'test'
        '''
        super().__init__(name, mode, **kargs)

    # frame[0] = image
    # frame[1] = joint action
    # frame[2] = joint observation
    # frame[3] = gripper joint observation

    def get_action(self, episode):
        ao = self.get_arm_joint_action(episode)
        go = self.get_gripper_joint_action(episode)
        return np.concatenate([ao, go], axis=1)

    def get_time_labels(self, episode, start_from_zero=True):
        tms = np.array([f[1].header.stamp.to_sec() for f in episode])
        if start_from_zero:
            return tms - tms[0]
        else:
            return tms

    def reorder_joint_values(self, joint_names, joint_positions):
        d = dict(zip(joint_names, joint_positions))
        return [d[jo] for jo in URDataset.arm_joint_order]         

    def get_arm_joint_observation(self, episode):
        return np.array([self.reorder_joint_values(f[2].name, f[2].position) for f in episode])

    def get_arm_joint_action(self, episode):
        return np.array([self.reorder_joint_values(f[1].joint_names, f[1].points[0].positions) for f in episode])

    def get_gripper_joint_action(self, episode):  # observation = action in gripper
        return np.array([f[3].position for f in episode])

    def load_group(self, group, load_image=True, image_size=(240, 320),
                   sampling_interval=1, normalize=True):
        path = os.path.join(self.dataset_path, '%s/%d' % (self._directory, group))

        state_file = os.path.join(path, 'states.pkl')
        episode = pd.read_pickle(state_file)
        joint_seq = self.get_action(episode)

        if normalize:
            jmin = self.joint_min_positions
            jmax = self.joint_max_positions
            joint_seq = (joint_seq - jmin) / (jmax - jmin)

        # load images
        n_frames = joint_seq.shape[0]
        if load_image:
            frames = []
            for i in range(0, n_frames, sampling_interval):
                img = plt.imread(os.path.join(path, 'image_frame%05d.jpg' % i))
                img = cv2.resize(img, (image_size[1], image_size[0]))
                if normalize:
                    img = img/255.
                frames.append(img)

            return joint_seq, frames
        else:
            return joint_seq

    def resample_episodes(self, threshold=1e-3):
        for i, e in enumerate(self.data):
            print(f'resampling episode {i}')
            output_js, output_imgs = self.resample_episode(e, threshold=threshold)
            group_dir = os.path.join('data', str(i))
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)
            np.savetxt(os.path.join(group_dir, 'joint_position.txt'), output_js)
            for j, img in enumerate(output_imgs):
                plt.imsave(os.path.join(group_dir, f'image_frame{j:05d}.jpg'), img)

    def resample_episode(self, episode, threshold=1e-3):
        js, imgs = episode
        output_js = [js[0]]
        output_imgs = [imgs[0]]
        l = 0.0
        for i in range(1, len(js)):
            l += np.linalg.norm(js[i-1] - js[i], ord=1)
            if l > threshold:
                output_js.append(js[i])
                output_imgs.append(imgs[i])
                l = 0.0
        return output_js, output_imgs


ds = URDataset('pen-kitting-real', 'train')
ds.load(image_size=(240, 320), groups=range(48), normalize=False)
ds.resample_episodes()


# def read_episode(episode_number):
#     return pd.read_pickle(os.path.join(os.environ['HOME'],
#                                        'Dataset/dataset2/pen-kitting-real230508',
#                                        str(episode_number),
#                                        'states.pkl'))

# episodes = [read_episode(n) for n in range(48)]
# episode = episodes[2]

def plot_action(episode, normalize=True):
    tms = get_time_labels(episode)
    a = get_action(episode)
    if normalize:
        a = normalize_joint_values(a)
    fig, ax = plt.subplots(facecolor="w")
    for i in range(7):
        ax.plot(tms, a[:, i], label=f'joint{i}')
    ax.legend()
    ax.set_xlabel('time[sec]')
    ax.set_ylabel('joint angle[normalized]')
    plt.show()


def plot_action_and_observation(episode, joint_number=0, normalize=True):
    tms = get_time_labels(episode)
    a = get_action(episode)
    if normalize:
        a = normalize_joint_values(a)
    fig, ax = plt.subplots(facecolor="w")
    ax.plot(tms, a[:, joint_number], label=f'action joint{joint_number}')
    o = get_arm_joint_observation(episode)
    if normalize:
        o = normalize_joint_values(o)
    ax.plot(tms, o[:, joint_number], label=f'observation joint{joint_number}')
    ax.legend()
    ax.set_xlabel('time[sec]')
    ax.set_ylabel('joint angle[normalized]')
    plt.show()

def compute_joint_range(episodes):
    maxs = []
    mins = []
    for e in episodes:
        a = get_action(e)
        maxs.append(np.max(a, axis=0))
        mins.append(np.min(a, axis=0))
    return np.min(np.array(mins), axis=0), np.max(np.array(maxs), axis=0)

def normalize_joint_values(joint_values):
    if joint_values.shape[1] < 7:
        l = joint_values.shape[1]
        return (joint_values - mins[:l]) / (maxs[:l] - mins[:l])
    else:
        return (joint_values - mins) / (maxs - mins)


# check synchronization accuracy

# check trajectory smoothness (arm + gripper observation)

# check arm joint delay (compare action/observation for single arm joint)

# gen input data for training a model