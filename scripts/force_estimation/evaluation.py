# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
import pandas as pd
import forcemap
from force_estimation_data_loader import ForceEstimationDataLoader
import force_estimation_v2_1 as fe
import force_distribution_viewer

import torch

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


bottom, wall, mid_air = pd.read_pickle('voxels.pkl')


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


def recall(evaluation_points, f_label, f, eps_true=1e-2, eps_detect=5e-3):
    label_points = 0
    recalled_points = 0
    for i,p in evaluation_points:
        if f_label[i] > eps_true:
            label_points += 1
            if f[i] > eps_detect:
                recalled_points += 1
    return label_points, recalled_points, recalled_points/label_points

def evaluate_recall(evaluation_points, n, eps_true=1e-2, eps_detect=5e-3):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    f_label = test_data[1][n]
    fmap.set_values(y_pred)
    y_pred = fmap.V
    fmap.set_values(f_label)
    f_label = fmap.V
    return recall(evaluation_points, f_label, y_pred, eps_true=eps_true, eps_detect=eps_detect)

def recall_all(evaluation_points, eps_true=[6e-3, 7e-3, 8e-3, 9e-3, 10e-3], eps_detect=5e-3):
    total_labels = np.zeros(len(eps_true))
    total_recalls = np.zeros(len(eps_true))
    for n in range(len(test_data[0])):
        print(n)
        for i in range(len(total_labels)):
            label_points, recalled_points, recall_ratio = evaluate_recall(evaluation_points, n, eps_true=eps_true[i], eps_detect=eps_detect)
            total_labels[i] += label_points
            total_recalls[i] += recalled_points
    return total_labels, total_recalls, total_recalls/total_labels

def precision(evaluation_points, f_label, f, eps_true=5e-3, eps_detect=1e-2):
    label_points = 0
    precise_points = 0
    for i,p in evaluation_points:
        if f_label[i] < eps_true:
            label_points += 1
            if f[i] < eps_detect:
                precise_points += 1
    return label_points, precise_points, precise_points/label_points

def evaluate_precision(evaluation_points, n, eps_true=5e-3, eps_detect=1e-2):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    f_label = test_data[1][n]
    fmap.set_values(y_pred)
    y_pred = fmap.V
    fmap.set_values(f_label)
    f_label = fmap.V
    return precision(evaluation_points, f_label, y_pred, eps_true=eps_true, eps_detect=eps_detect)

def precision_all(evaluation_points, eps_true=5e-3, eps_detect=[6e-3, 7e-3, 8e-3, 9e-3, 10e-3]):
    total_labels = np.zeros(len(eps_detect))
    total_precises = np.zeros(len(eps_detect))
    for n in range(len(test_data[0])):
        print(n)
        for i in range(len(total_labels)):
            label_points, precise_points, recall_ratio = evaluate_precision(evaluation_points, n, eps_true=eps_true, eps_detect=eps_detect[i])
            total_labels[i] += label_points
            total_precises[i] += precise_points
    return total_labels, total_precises, total_precises/total_labels
    

bottom_recall = (np.array([1550983., 1493301., 1441067., 1393395., 1349422.]),
                 np.array([1374809., 1338128., 1303116., 1269411., 1237443.]),
                 np.array([0.88641139, 0.89608726, 0.90427163, 0.91102021, 0.91701706]))

wall_recall = (np.array([1320601., 1240626., 1170670., 1108684., 1053952.]),
               np.array([957979., 920726., 885488., 852148., 821310.]),
               np.array([0.72541138, 0.7421463 , 0.7563942 , 0.76861216, 0.77926699]))

mid_air_recall = (np.array([1522196., 1402607., 1299813., 1211266., 1132615.]),
                  np.array([1073135., 1015101.,  961070.,  911598.,  865422.]),
                  np.array([0.70499134, 0.72372446, 0.73939097, 0.75259935, 0.76409195]))

bottom_precision = (np.array([1623675., 1623675., 1623675., 1623675., 1623675.]),
                    np.array([1346230., 1383512., 1413263., 1437948., 1458415.]),
                    np.array([0.82912529, 0.85208678, 0.87041003, 0.88561319, 0.89821855]))

wall_precision = (np.array([7639667., 7639667., 7639667., 7639667., 7639667.]),
                  np.array([7326094., 7383636., 7427007., 7460903., 7487408.]),
                  np.array([0.95895462, 0.96648663, 0.97216371, 0.97660055, 0.98006994]))

mid_air_precision = (np.array([24236007., 24236007., 24236007., 24236007., 24236007.]),
                     np.array([23811556., 23892200., 23953374., 24000946., 24038249.]),
                     np.array([0.98248676, 0.98581421, 0.9883383 , 0.99030117, 0.99184032]))


import matplotlib.pyplot as plt

def plot_recall_curve():
    xs = [6e-3, 7e-3, 8e-3, 9e-3, 10e-3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('recall')
    ax.set_ylabel('detection rate')
    ax.set_xlabel('epsilon true [Ns/m^3]')
    ax.plot(xs, bottom_recall[2], label='contact with the bottom')
    ax.plot(xs, wall_recall[2], label='contact with the wall')
    ax.plot(xs, mid_air_recall[2], label='contact between objects')
    ax.legend(loc='best')
    ax.set_ylim(0.5, 1)
    ax.set_xticks([6e-3, 7e-3, 8e-3, 9e-3, 1e-2])
    plt.show()

def plot_precision_curve():
    xs = [6e-3, 7e-3, 8e-3, 9e-3, 10e-3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('precision')
    ax.set_ylabel('detection rate')
    ax.set_xlabel('epsilon detect [Ns/m^3]')
    ax.plot(xs, bottom_precision[2], label='contact with the bottom')
    ax.plot(xs, wall_precision[2], label='contact with the wall')
    ax.plot(xs, mid_air_precision[2], label='contact between objects')
    ax.legend(loc='best')
    ax.set_ylim(0.5, 1)
    ax.set_xticks([6e-3, 7e-3, 8e-3, 9e-3, 1e-2])
    plt.show()


def f(n):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    f_label = test_data[1][n]
    return y_pred, f_label, np.sum(np.abs(y_pred - f_label)) / np.sum(f_label)


def KL(p, q):
    if type(p) != torch.Tensor:
        p = torch.Tensor(p) 
    if type(q) != torch.Tensor:
        q = torch.Tensor(q)
    p = p / p.sum()
    q = q / q.sum()
    return (p * (p / q).log()).sum()
