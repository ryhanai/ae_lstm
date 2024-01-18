import functools
import operator
import os
import time
from concurrent import futures
from pathlib import Path

import forcemap
import matplotlib.pyplot as plt
import mesh2sdf
import numpy as np
import pandas as pd
import trimesh
from force_estimation import force_distribution_viewer
from scipy.spatial.transform import Rotation as R
from sim_isaac.object_loader import ObjectInfo

fmap = forcemap.GridForceMap("small_table", bandwidth=0.03)
# viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


# data_dir = f"{os.environ['HOME']}/Dataset/dataset2/basket-filling230918"
# obj_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/specification/meshes/objects"
data_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/scripts/sim_isaac/data"
obj_dir = f"{os.environ['HOME']}/Dataset/ycb_conveni"
config_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/specification/config"
obj_info = ObjectInfo(obj_dir, config_dir, "ycb_conveni_v1")

# table coordinate
# ([0,0,0.72],[0,0,0,1]), scale = [1,1,0.01]


def compute_sdf(mesh, state, size, scale):
    p, q = state
    r = R.from_quat(q)
    # r.as_matrix()

    vertices = r.apply(mesh.vertices) + p - np.array([0, 0, 0.73 + 0.10])
    vertices = vertices * scale

    sdf = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=2 / size, return_mesh=False)
    # m.export('hoge.obj')
    df = np.where(sdf >= 0, sdf, 0)
    return df


def normal_distribution(fmap, contact_position):
    mean = contact_position
    sigma = fmap.bandwidth
    v = [np.exp(-(((np.linalg.norm(p - mean)) / sigma) ** 2)) for p in fmap.positions]
    v = np.array(v).reshape(fmap.get_grid_shape())
    return v


def get_obj_file(name):
    """
    return obj file name and scale
    """
    p = Path(self._object_info.usd_file(name))
    if obj_info.dataset(name) == "ycb":
        obj_file = p.parent / f"{name}.obj"
    if obj_info.dataset(name) == "conveni":
        obj_file = p.parent.parent / f"{name}.obj"
    scale = 1.0
    return obj_file, scale

    # if name == "seria_basket":
    #     return f"{obj_dir}/seria_basket_body_collision.obj", 0.001
    # else:
    #     return f"{obj_dir}/ycb/{name}/google_16k/textured.obj", 1.0


def get_obj_position(bin_state, object_name):
    try:
        return [o[1] for o in bin_state if o[0] == object_name][0]
    except IndexError:
        return (np.array([0, 0, 0.73]), R.from_euler("xyz", [0, 0, 1.57080], degrees=False).as_quat())


def compute_density(bin_state, contacts, size=60, scale=1 / 0.2, sigma_d=0.01):
    return do_compute_density(bin_state, contacts, size, scale, sigma_d)


def do_compute_density(bin_state, contacts, size=60, scale=1 / 0.2, sigma_d=0.01):
    sdfs = [np.zeros((size, size, size))]
    f_dists = [np.zeros((size, size, size))]
    weighted_f_dists = [np.zeros((size, size, size))]

    # compute SDFs for all the objects in bin_state

    for contact_position, force_value, contact_pair in zip(*contacts):
        objectA, objectB = contact_pair
        print(objectA, objectB)

        obj_file1, mesh_scale1 = get_obj_file(objectA)
        obj_file2, mesh_scale2 = get_obj_file(objectB)
        mesh1 = trimesh.load(obj_file1, force="mesh")
        mesh1.vertices = mesh_scale1 * mesh1.vertices
        mesh2 = trimesh.load(obj_file2, force="mesh")
        mesh2.vertices = mesh_scale2 * mesh2.vertices

        # contact force distribution
        g = normal_distribution(fmap, contact_position)

        # geometry-guided weight for objectA
        s1 = get_obj_position(bin_state, objectA)
        sdf1 = compute_sdf(mesh1, s1, size=size, scale=scale)
        esdf1 = np.exp(-sdf1 / (sigma_d * scale))

        # geometry weight for objectB
        s2 = get_obj_position(bin_state, objectB)
        sdf2 = compute_sdf(mesh2, s2, size=size, scale=scale)
        esdf2 = np.exp(-sdf2 / (sigma_d * scale))

        # return edf1, edf2, g
        # kde_dists.append(g)

        # geometry potential
        sdfs.append(np.min(sdf1, sdf2))

        # force distribution without geometry weight
        f_dist = force_value * g
        f_dists.append(f_dist)

        unnormalized_wkde = edf1 * edf2 * g
        alpha = 1.0 / max(np.sum(unnormalized_wkde), 1e-3)
        normalized_wkde = alpha * force_value * unnormalized_wkde
        weighted_dists.append(normalized_wkde)

    force_distribution = functools.reduce(operator.add, f_dists)
    weighted_force_distribution = functools.reduce(operator.add, kde_dists), functools.reduce(
        operator.add, weighted_dists
    )
    return

    return functools.reduce(operator.add, weighted_dists)


# def do_compute_density(bin_state, contacts, size=60, scale=1 / 0.2, sigma_d=0.01):
#     sdfs = [np.zeros((size, size, size))]
#     f_dists = [np.zeros((size, size, size))]
#     weighted_f_dists = [np.zeros((size, size, size))]

#     for contact_position, force_value, contact_pair in zip(*contacts):
#         objectA, objectB = contact_pair
#         print(objectA, objectB)

#         obj_file1, mesh_scale1 = get_obj_file(objectA)
#         obj_file2, mesh_scale2 = get_obj_file(objectB)
#         mesh1 = trimesh.load(obj_file1, force="mesh")
#         mesh1.vertices = mesh_scale1 * mesh1.vertices
#         mesh2 = trimesh.load(obj_file2, force="mesh")
#         mesh2.vertices = mesh_scale2 * mesh2.vertices

#         # contact force distribution
#         g = normal_distribution(fmap, contact_position)

#         # geometry-guided weight for objectA
#         s1 = get_obj_position(bin_state, objectA)
#         sdf1 = compute_sdf(mesh1, s1, size=size, scale=scale)
#         esdf1 = np.exp(-sdf1 / (sigma_d * scale))

#         # geometry weight for objectB
#         s2 = get_obj_position(bin_state, objectB)
#         sdf2 = compute_sdf(mesh2, s2, size=size, scale=scale)
#         esdf2 = np.exp(-sdf2 / (sigma_d * scale))

#         # return edf1, edf2, g
#         # kde_dists.append(g)

#         # geometry potential
#         sdfs.append(np.min(sdf1, sdf2))

#         # force distribution without geometry weight
#         f_dist = force_value * g
#         f_dists.append(f_dist)

#         unnormalized_wkde = edf1 * edf2 * g
#         alpha = 1.0 / max(np.sum(unnormalized_wkde), 1e-3)
#         normalized_wkde = alpha * force_value * unnormalized_wkde
#         weighted_dists.append(normalized_wkde)

#     force_distribution = functools.reduce(operator.add, f_dists)
#     weighted_force_distribution = functools.reduce(operator.add, kde_dists), functools.reduce(
#         operator.add, weighted_dists
#     )
#     return

#     return functools.reduce(operator.add, weighted_dists)


def compute_force_distribution(frameNo, log_scale=False, overwrite=False):
    out_file = os.path.join(data_dir, "force_zip{:05d}.pkl".format(frameNo))
    if (not overwrite) and os.path.exists(out_file):
        print(f"skip [{frameNo}]")
    else:
        print(f"process [{frameNo}], log_scale={log_scale}")
        bin_state = pd.read_pickle(f"{data_dir}/bin_state{frameNo:05}.pkl")
        contacts = pd.read_pickle(f"{data_dir}/contact_raw_data{frameNo:05}.pkl")
        d = compute_density(bin_state, contacts)
        if log_scale:
            d = np.log(1 + d)
        pd.to_pickle(d, out_file)


def compute_force_distribution_for_all(scene_numbers=range(0, 2000)):
    start_tm = time.time()
    with futures.ProcessPoolExecutor() as executor:
        executor.map(compute_force_distribution, scene_numbers)
    print(f"compute force distribution took: {time.time() - start_tm} [sec]")


def visualize(bin_state, d, scale=0.4):
    fmap.set_values(d * scale)
    viewer.publish_bin_state(bin_state, fmap)


# compute_force_distribution_for_all()
