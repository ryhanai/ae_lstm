# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import quaternion
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
test_data_real = dl.load_real_data_for_rgb2fmap(test_mode=True)
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


def f(y_pred, object_center=[0.02, -0.04, 0.79], object_radius=0.05):
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = y_pred
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fmap.get_positions()
    idx = np.where(scipy.linalg.norm(ps - object_center, axis=1) < object_radius)[0]
    ps = ps[idx]
    g_vecs = g_vecs[idx]
    idx = np.where(np.sum((ps - object_center) * g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]
    return fps, fg_vecs

# sim
pick_example = [
    (25, [0.03, -0.03, 0.78]),
    (14, [0.01, 0.03, 0.79]),
    (24, [0.03, 0.06, 0.76]),
    (27, [0.03, -0.03, 0.76]),
    (28, [0.01, -0.03, 0.775]),
    (36, [0.005, -0.015, 0.775])
]
# real
pick_example_real = [
    (1, [0.01, 0.02, 0.82]),
    (7, [0.02, -0.075, 0.78]),  # maybe best
    (8, [-0.03, 0.08, 0.78]),  # bimyo-
    (9, [0.004, -0.07, 0.77]),  # need rotation
    (10, [-0.01, 0.06, 0.75]),  # need rotation
    (14, [0.035, 0.015, 0.77]),
    (15, [-0.01, -0.01, 0.74]),  # need rotation
]

# cons = (
#     {'type': 'eq', 'fun': lambda x: scipy.linalg.norm(x) - 1}
# )
cons = (
    {'type': 'eq', 'fun': lambda x: scipy.linalg.norm(x[:3]) - 1},
    {'type': 'eq', 'fun': lambda x: scipy.linalg.norm(x[3:6]) - 1},
    {'type': 'eq', 'fun': lambda x: np.abs(x[6])},
)


def LeakyReLU(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)


def f_target(fps,
             fg_vecs,
             alpha,
             c,  # object center
             x):
    v = x[:3]
    rot_axis = x[3:6]
    omega = x[6]
    neg_fg_vecs = -fg_vecs
    dp = v + np.cross(omega * rot_axis, fps - c)
    return np.sum(LeakyReLU(np.sum(neg_fg_vecs * dp, axis=1), alpha=alpha))


def pick_direction_plan(y_pred, bin_state, object_center, object_radius, scale=[0.005, 0.02, 0.01], alpha=1.0):
    fps, fg_vecs = f(y_pred, object_center, object_radius)
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=False, draw_force_gradient=False)
    viewer.draw_vector_field(fps, fg_vecs, scale=0.3)
    viewer.rviz_client.draw_sphere(object_center, [1, 0, 0, 1], [0.01, 0.01, 0.01])
    viewer.rviz_client.show()

    def f_objective(x):
        return f_target(fps, fg_vecs, alpha, object_center, x=x)

    result = minimize(f_objective, x0=np.array([0, 0, 1, 0, 0, 1, 0.2]), constraints=cons, method='SLSQP')
    print(result)
    pick_direction = result.x[:3]
    pick_rot_axis = result.x[3:6]
    pick_omega = result.x[6]

    viewer.rviz_client.draw_arrow(object_center, object_center + pick_direction * 0.1, [1, 0, 1, 1], scale)
    viewer.rviz_client.draw_arrow(object_center, object_center + pick_rot_axis * pick_omega * 0.1, [1, 1, 0, 1], [0.002, 0, 0])

    ez = pick_rot_axis * np.sign(pick_omega)
    ey = np.cross([1, 0, 0], ez)
    ex = np.cross(ey, ez)
    q = quaternion.from_rotation_matrix(np.transpose(np.stack([ex, ey, ez])))
    quat = np.array([q.x, q.y, q.z, q.w])
    viewer.rviz_client.draw_mesh("package://force_estimation/meshes_extra/rotation.dae",
                                 (object_center, quat),
                                 (1., 1., 0., 1.),
                                 0.1 + 0.6 * np.abs(pick_omega) * np.array([1., 1., 1.]))

    viewer.rviz_client.show()
    return fps, fg_vecs, pick_direction, pick_rot_axis, pick_omega


def pick_direction_plan_sim(n=25, object_center=[0.03, -0.02, 0.78], object_radius=0.05, alpha=1.0):
    y_pred = model.predict(test_data[0][n:n+1])[0]
    bin_state = test_data[2][n]
    return pick_direction_plan(y_pred, bin_state, object_center=object_center, object_radius=object_radius, alpha=alpha)


def pick_direction_plan_real(n=1, object_center=[0.03, -0.02, 0.78], object_radius=0.05, alpha=1.0, show_rgb=True):
    y_pred = model.predict(test_data_real[n:n+1])[0]
    result = pick_direction_plan(y_pred, None, object_center=object_center, object_radius=object_radius, alpha=alpha)
    if show_rgb:
        plt.imshow(test_data_real[n])
        plt.show()
    return result


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


