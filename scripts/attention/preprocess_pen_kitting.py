# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


episode_number = 1
episode = pd.read_pickle(
    os.path.join(os.environ['HOME'],
                 'Dataset/dataset2/pen-kitting-real23-5-2',
                 str(episode_number),
                 'states.pkl')
)


def parse(episode):
    frame = episode[0]
    img, joint_command, arm_joint_positions, gripper_joint_positions = frame


def get_time_labels(episode):


def reorder_joint_values(joint_positions):
    pass


def get_joint_observation(episode):
    obs = [f[2].position[2] for f in episode]


def get_joint_action(episode):
    tms = [f[1].header.stamp.to_sec() for f in episode]
    act = [reorder_joint_values(f[1].points[0].positions[0]) for f in episode]
    # plt.plot(tms, act[:,0])
    # plt.plot(tms, act[:,1])
    # plt.show()



# check synchronization accuracy

# check trajectory smoothness (arm + gripper observation)

# check arm joint delay (compare action/observation for single arm joint)

# gen input data for training a model

