# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
import forcemap
from force_estimation_data_loader import ForceEstimationDataLoader
import force_estimation_v2_1 as fe
import force_distribution_viewer


dataset = 'basket-filling2'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]

fmap = forcemap.GridForceMap('seria_basket')

dl = ForceEstimationDataLoader(os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset),
                               os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset+'-real'),
                               image_height=image_height,
                               image_width=image_width,
                               start_seq=1,
                               n_seqs=1800,  # n_seqs=1500,
                               start_frame=3, n_frames=3,
                               real_start_frame=1, real_n_frames=294
                               )

model = fe.model_rgb_to_fmap_res50()
model.load_weights('../../runs/ae_cp.basket-filling2.model_resnet.20221202165608/cp.ckpt')
test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


def f_diff(t, y):
    return np.abs(t - y)

def plot_map(f, n):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    force_label = test_data[1][n]
    z = f(force_label, y_pred)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = z
    fmap.set_values(fv)
    bin_state = test_data[2][n]
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=True, draw_force_gradient=False)

def shoot_rays(p, directions):
    return [S.p.rayTest(p, p + d)[0][3] for d in directions]

def collect_bottom_voxels(voxels, epsilon=1e-2):
    bottom_voxels = []
    for i, p in enumerate(voxels):
        if abs(p[2] - 0.73) < epsilon:
            bottom_voxels.append((i, p))
    return bottom_voxels

def collect_wall_voxels(voxels, epsilon=1e-2):
    wall_voxels = []
    for i, p in enumerate(voxels):
        cps = shoot_rays(p, directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]))
        distance = min(scipy.linalg.norm(cps - p, axis=1))
        print(distance)
        if distance < epsilon:
            wall_voxels.append((i, p))
    return wall_voxels

def collect_mid_air_voxels(voxels, epsilon=2e-2):
    mid_air_voxels = []
    for i, p in voxels:
        cps = shoot_rays(p, directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)]))
        distance = min(scipy.linalg.norm(cps - p, axis=1))
        print(distance)
        if distance > epsilon:
            mid_air_voxels.append((i, p))
    return mid_air_voxels

def separate_voxels():
    bottom_voxels = collect_bottom_voxels(voxels)
    wall_voxels = collect_wall_voxels(voxels)
    mid_air_voxels = collect_mid_air_voxels(voxels)

def recall(evaluation_points, f_label, f, epsilon=1e-3):
    label_points = 0
    recalled_points = 0
    for i,p in evaluation_points:
        if f_label[i] > epsilon:
            label_points += 1
            if f[i] > epsilon:
                recalled_points += 1
    return label_points, recalled_points, recalled_points/label_points

def recall_curve(evaluation_points, f_lable, f):
    f_targets = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    for f_target in f_targets:
        recalls = recall(evaluation_points, f_label, f, f_target)
    return recalls

