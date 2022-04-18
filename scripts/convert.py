# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import cv2

data_dir = os.path.join(os.environ['HOME'], 'Dataset/dataset2/reaching-real-raw')
joint_names = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
    ]

def convert(group=5, save_threshold=2e-2):
    data = pd.read_pickle(os.path.join(data_dir, str(group), 'states.pkl'))

    frameNo = 0
    previous_js = np.zeros(7)
    frames = []

    for i,frame in enumerate(data):
        command = frame[2]
        b = dict(zip(command.joint_names, command.points[0].positions))
        js = np.array([b[k] for k in joint_names] + [0])
        tm = command.header.stamp.to_time()
        if np.linalg.norm(js - previous_js, ord=1) > save_threshold:
            print('save:[{}]: {}'.format(frameNo, js))
            img = cv2.imread(os.path.join(data_dir, str(group), 'image_frame%05d.jpg'%i))
            d = {'frameNo':frameNo, 'jointPosition':js, 'image':img}
            frames.append(d)
            frameNo += 1
            previous_js = js

    output_dir = os.path.join('converted_data', str(group))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joint_positions = []
    for d in frames:
        frameNo = d['frameNo']
        joint_positions.append(d['jointPosition'])
        cv2.imwrite(os.path.join(output_dir, 'image_frame%05d.jpg'%frameNo), d['image'])
    np.savetxt(os.path.join(output_dir, 'joint_position.txt'), joint_positions)
