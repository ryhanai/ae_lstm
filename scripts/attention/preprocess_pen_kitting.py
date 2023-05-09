# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_episode(episode_number):
    return pd.read_pickle(os.path.join(os.environ['HOME'],
                                       'Dataset/dataset2/pen-kitting-real230508',
                                       str(episode_number),
                                       'states.pkl'))


episodes = [read_episode(n) for n in range(48)]
episode = episodes[2]


arm_joint_order = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


# frame[0] = image
# frame[1] = joint action
# frame[2] = joint observation
# frame[3] = gripper joint observation


def get_time_labels(episode, start_from_zero=True):
    tms = np.array([f[1].header.stamp.to_sec() for f in episode])
    if start_from_zero:
        return tms - tms[0]
    else:
        return tms


def reorder_joint_values(joint_names, joint_positions):
    d = dict(zip(joint_names, joint_positions))
    return [d[jo] for jo in arm_joint_order]         


def get_arm_joint_observation(episode):
    return np.array([reorder_joint_values(f[2].name, f[2].position) for f in episode])


def get_arm_joint_action(episode):
    return np.array([reorder_joint_values(f[1].joint_names, f[1].points[0].positions) for f in episode])


def get_gripper_joint_action(episode):  # observation = action in gripper
    return np.array([f[3].position for f in episode])


def get_action(episode):
    ao = get_arm_joint_action(episode)
    go = get_gripper_joint_action(episode)
    return np.concatenate([ao, go], axis=1)


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


mins, maxs = compute_joint_range(episodes)

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