# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
from force_estimation import force_estimation_v2_1 as fe
from force_estimation import force_distribution_viewer
from force_estimation import forcemap
from force_estimation.force_estimation_data_loader import ForceEstimationDataLoader

# from core import dataset
# from sim import run_sim_basket_filling

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


def plan():
    pass


def load_scene():
    pass


def pick():
    pass


def compute_disturbance():
    pass


def evaluate_algorithm():
    load_scene()
    plan()
    pick()
    compute_disturbance()


model_file = 'ae_cp.basket-filling2.model_resnet.20221202165608'

model = fe.model_rgb_to_fmap_res50()
model.load_weights('../../runs/ae_cp.basket-filling2.model_resnet.20221202165608/cp.ckpt')
test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


def show_bin_state(bin_state, force_values, draw_fmap=True, draw_force_gradient=False):
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = force_values
    fmap.set_values(fv)
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=draw_fmap, draw_force_gradient=draw_force_gradient)


def pick_direction_plan_old(n=25, gp=[0.02, -0.04, 0.79], radius=0.05, scale=[0.005, 0.01, 0.004]):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    
    force_label = test_data[1][n]
    bin_state = test_data[2][n]

    gp = np.array(gp)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = y_pred
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fmap.get_positions()
    idx = np.where(np.sum((ps - gp)*g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]

    # points for visualization
    filtered_pos_val_pairs = [(p, g) for (p, g) in zip(fps, fg_vecs) if scipy.linalg.norm(g) > 0.008]
    pz, vz = zip(*filtered_pos_val_pairs)
    pz = np.array(pz)
    vz = np.array(vz)
    fmap.set_values(ps)
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=False, draw_force_gradient=False)
    viewer.draw_vector_field(pz, vz, scale=0.3)
    viewer.rviz_client.draw_sphere(gp, [1, 0, 0, 1], [0.01, 0.01, 0.01])
    viewer.rviz_client.show()

    # points for planning
    pz = fps
    vz = fg_vecs
    idx = np.where(scipy.linalg.norm(pz - gp, axis=1) < radius)[0]
    pz = pz[idx]
    vz = vz[idx]
    pick_direction = np.sum(vz, axis=0)
    pick_direction /= np.linalg.norm(pick_direction)
    viewer.rviz_client.draw_arrow(gp, gp + pick_direction * 0.1, [0, 1, 0, 1], scale)
    pick_moment = np.sum(np.cross(pz - gp, vz), axis=0)
    pick_moment /= np.linalg.norm(pick_moment)
    viewer.rviz_client.draw_arrow(gp, gp + pick_moment * 0.1, [1, 1, 0, 1], scale)

    viewer.rviz_client.show()
    return pz, vz, pick_direction, pick_moment


def f(n=25, object_center=[0.02, -0.04, 0.79], object_radius=0.05):
    y_pred = model.predict(test_data[0][n:n+1])[0]

    object_center = np.array(object_center)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = y_pred
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fmap.get_positions()
    idx = np.where(np.sum((ps - object_center) * g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]
    return fps, fg_vecs


def pick_direction_plan(n=25, object_center=[0.02, -0.04, 0.79], object_radius=0.05, scale=[0.005, 0.01, 0.004]):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    bin_state = test_data[2][n]

    gp = np.array(gp)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = y_pred
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fmap.get_positions()
    idx = np.where(np.sum((ps - gp)*g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]

    # points for visualization
    filtered_pos_val_pairs = [(p, g) for (p, g) in zip(fps, fg_vecs) if scipy.linalg.norm(g) > 0.008]
    pz, vz = zip(*filtered_pos_val_pairs)
    pz = np.array(pz)
    vz = np.array(vz)

    # points for planning
    pz = fps
    vz = fg_vecs
    idx = np.where(scipy.linalg.norm(pz - gp, axis=1) < radius)[0]
    pz = pz[idx]
    vz = vz[idx]

    pick_direction = np.sum(vz, axis=0)
    pick_direction /= np.linalg.norm(pick_direction)

    pick_moment = np.sum(np.cross(pz - gp, vz), axis=0)
    pick_moment /= np.linalg.norm(pick_moment)

    fmap.set_values(ps)
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=False, draw_force_gradient=False)
    viewer.draw_vector_field(pz, vz, scale=0.3)
    viewer.rviz_client.draw_sphere(gp, [1, 0, 0, 1], [0.01, 0.01, 0.01])
    viewer.rviz_client.show()

    viewer.rviz_client.draw_arrow(gp, gp + pick_direction * 0.1, [0, 1, 0, 1], scale)
    viewer.rviz_client.draw_arrow(gp, gp + pick_moment * 0.1, [1, 1, 0, 1], scale)

    viewer.rviz_client.show()
    return pz, vz, pick_direction, pick_moment


def virtual_pick(bin_state0, pick_vector, pick_moment, object_name='011_banana', alpha=0.01, beta=0.05, repeat=5):
    def do_virtual_pick():
        bin_state = bin_state0
        for i in range(10):
            p, q = [s for s in bin_state if s[0] == object_name][0][1]
            dq = quat_from_euler(beta * pick_moment)
            dp = alpha * pick_vector
            p2, q2 = multiply_transforms(dp, dq, p, q)
            p2 = p + dp
            bin_state = [(object_name, (p2, q2)) if s[0] == object_name else s for s in bin_state]
            viewer.publish_bin_state(bin_state, [], [], draw_fmap=False, draw_force_gradient=False)
            time.sleep(0.2)

    for i in range(repeat):
        do_virtual_pick()
