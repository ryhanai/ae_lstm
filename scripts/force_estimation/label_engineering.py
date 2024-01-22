import functools
import operator
import os
import time
from concurrent import futures

import forcemap
import matplotlib.pyplot as plt
import mesh2sdf
import numpy as np
import pandas as pd
import trimesh
from core.object_loader import ObjectInfo
from core.utils import message
from force_estimation import force_distribution_viewer
from scipy.spatial.transform import Rotation as R

fmap = forcemap.GridForceMap("small_table", bandwidth=0.03)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


data_dir = f"{os.environ['HOME']}/Dataset/forcemap/tabletop240121"
object_info = ObjectInfo("ycb_conveni_v1")

# table coordinate
# ([0,0,0.72],[0,0,0,1]), scale = [1,1,0.01]


def load(scene_number):
    bin_state = pd.read_pickle(f"{data_dir}/bin_state{scene_number:05d}.pkl")
    contact_state = pd.read_pickle(f"{data_dir}/contact_raw_data{scene_number:05d}.pkl")
    return bin_state, contact_state


def get_obj_position(bin_state, object_name):
    try:
        return [o[1] for o in bin_state if o[0] == object_name][0]
    except IndexError:
        return (np.array([0, 0, 0.73]), R.from_euler("xyz", [0, 0, 1.57080], degrees=False).as_quat())


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


def sdfs_for_objects(bin_state, scale=1 / 0.2, sigma_d=0.01):
    x, y, z = fmap.get_grid_shape()
    size = max(x, y)
    height = z
    sdfs = {}
    for object_name, object_pose in bin_state:
        obj_file, mesh_scale = object_info.obj_file(object_name)
        mesh = trimesh.load(obj_file, force="mesh")
        mesh.vertices = mesh_scale * mesh.vertices
        sdf = compute_sdf(mesh, object_pose, size=size, scale=scale)
        sdf = sdf[:, :, :height]
        sdfs[object_name] = sdf
    return sdfs


def sdf_for_scene(sdfs):
    return functools.reduce(lambda x, y: np.minimum(x, y), sdfs.values())


def compute_density(bin_state, contact_state, scale=1 / 0.2, sigma_d=0.01):
    start_t = time.time()
    sdfs = sdfs_for_objects(bin_state, scale, sigma_d)
    message(f"SDF computation: {time.time() - start_t:.2f}[sec], number of SDFs={len(sdfs)}")

    fdists = []
    weighted_fdists = []

    def get_sdf(object_name):
        return sdfs[object_name]

    start_t = time.time()
    for contact_position, force_value, contact_pair in zip(*contact_state):
        objectA, objectB = contact_pair
        if objectA == "table" or objectB == "table":
            continue
        # print(objectA, objectB)

        sdf1 = get_sdf(objectA)
        esdf1 = np.exp(-sdf1 / (sigma_d * scale))
        sdf2 = get_sdf(objectB)
        esdf2 = np.exp(-sdf2 / (sigma_d * scale))

        g = normal_distribution(fmap, contact_position)
        fdist = force_value * g
        fdists.append(fdist)

        unnormalized_wkde = esdf1 * esdf2 * g
        alpha = 1.0 / max(np.sum(unnormalized_wkde), 1e-3)
        normalized_wkde = alpha * force_value * unnormalized_wkde
        weighted_fdists.append(normalized_wkde)

    message(f"contact point process: {time.time() - start_t:.2f}[sec], number of contacs={len(contact_state[0])}")

    start_t = time.time()
    force_distribution = functools.reduce(operator.add, fdists)
    weighted_force_distribution = functools.reduce(operator.add, weighted_fdists)
    scene_sdf = sdf_for_scene(sdfs)
    message(f"post process: {time.time() - start_t:.2f}[sec]")

    return force_distribution, weighted_force_distribution, scene_sdf


# def do_compute_density(bin_state, contacts, size=60, scale=1 / 0.2, sigma_d=0.01):
#     sdfs = [np.zeros((size, size, size))]
#     f_dists = [np.zeros((size, size, size))]
#     weighted_f_dists = [np.zeros((size, size, size))]

#     # compute SDFs for all the objects in bin_state

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
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=True)


# compute_force_distribution_for_all()


if __name__ == "__main__":
    bin_state, contact_state = load(0)
    compute_density(bin_state, contact_state)
