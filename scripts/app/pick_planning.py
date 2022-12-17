# -*- coding: utf-8 -*-

import numpy as np
import force_estimation_v2 as fe


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

model_file = 'ae_cp.basket-filling2.model_resnet.20221202165608'

model = fe.model_rgb_to_fmap_res50()
test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
tester = Tester(model, test_data, model_file)


def show_bin_state(fcam, bin_state, fmap, draw_fmap=True, draw_force_gradient=False):
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = fmap
    positions = fcam.positions
    viewer.publish_bin_state(bin_state, positions, fv, draw_fmap=draw_fmap, draw_force_gradient=draw_force_gradient)


def pick_direction_plan(fcam, n=25, gp=[0.02, -0.04, 0.79], radius=0.05, scale=[0.005, 0.01, 0.004]):
    fmap, force_label, rgb = tester.predict_force_from_rgb(n, visualize_result=False)
    bin_state = tester.test_data[2][n]

    gp = np.array(gp)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = fmap
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fcam.positions
    idx = np.where(np.sum((ps - gp)*g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]

    # points for visualization
    filtered_pos_val_pairs = [(p, g) for (p, g) in zip(fps, fg_vecs) if scipy.linalg.norm(g) > 0.008]
    pz, vz = zip(*filtered_pos_val_pairs)
    pz = np.array(pz)
    vz = np.array(vz)
    viewer.publish_bin_state(bin_state, ps, fv, draw_fmap=False, draw_force_gradient=False)
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
