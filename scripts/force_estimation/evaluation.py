# -*- coding: utf-8 -*-

import os
import numpy as np
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
