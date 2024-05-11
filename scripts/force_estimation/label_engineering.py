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
from core.object_loader import ObjectInfo
from core.utils import message
from force_estimation import force_distribution_viewer
from mesh_to_sdf import mesh_to_voxels
from scipy.spatial.transform import Rotation as R

## config for T-RO '24, piling on the table, train with YCB+Conveni-v1
# data_dir = f"{os.environ['HOME']}/Dataset/forcemap/tabletop240304"
# object_info = ObjectInfo("ycb_conveni_v1")
# fmap = forcemap.GridForceMap("small_table", bandwidth=0.03)
# def in_forcemap_area(p):
#     return p[0] > -0.25 and p[0] < 0.25 and p[1] > -0.25 and p[1] < 0.25 and p[2] > 0.685 and p[2] < 1.05


## config for the latest basket scene
data_dir = f"{os.environ['HOME']}/Dataset/forcemap/basket240511"
object_info = ObjectInfo("ycb_conveni_v1_small")
fmap = forcemap.GridForceMap("seria_basket", bandwidth=0.03)

viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
viewer.set_object_info(object_info)


def in_forcemap_area(p):
    return p[0] > -0.18 and p[0] < 0.18 and p[1] > -0.18 and p[1] < 0.18 and p[2] > 0.70 and p[2] < 1.05


def in_forcemap_area_mesh2sdf(p):
    """need to remove objects outside of the forcemap strictly in case of mesh2sdf"""
    return p[0] > -0.2 and p[0] < 0.2 and p[1] > -0.2 and p[1] < 0.2 and p[2] > 0.735 and p[2] < 1.0


def load(scene_number, exclude_protruding_object=True):
    bin_state = pd.read_pickle(f"{data_dir}/bin_state{scene_number:05d}.pkl")
    contact_state = pd.read_pickle(f"{data_dir}/contact_raw_data{scene_number:05d}.pkl")
    if exclude_protruding_object:
        bin_state = [s for s in bin_state if in_forcemap_area(s[1][0])]
    return bin_state, contact_state


def get_obj_position(bin_state, object_name):
    return [o[1] for o in bin_state if o[0] == object_name][0]
    # try:
    #     return [o[1] for o in bin_state if o[0] == object_name][0]
    # except IndexError:
    #     return (np.array([0, 0, 0.73]), R.from_euler("xyz", [0, 0, 1.57080], degrees=False).as_quat())


def compute_sdf_with_mesh2sdf(mesh, state, fmap, zero_fill=False):
    # mesh_scale = 0.8
    mesh_scale = 1.0
    p, q = state
    r = R.from_quat(q)
    # r.as_matrix()
    vertices = r.apply(mesh.vertices) + p

    size = max(fmap.get_grid_shape())
    center = np.array([0, 0, fmap._zmin + fmap._zrange])
    scale = 2.0 * mesh_scale / max(fmap._xrange, fmap._yrange, fmap._zrange)
    vertices = (vertices - center) * scale

    sdf = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=2 / size, return_mesh=False)
    sdf = sdf / scale

    if zero_fill:  # fill zeros to the inside of the object
        return np.where(sdf >= 0, sdf, 0)
    else:
        return sdf


# def compute_sdf(mesh, state, fmap, zero_fill=False):
#     mesh_scale = 1.0
#     p, q = state
#     r = R.from_quat(q)
#     vertices = r.apply(mesh.vertices) + p

#     size = max(fmap.get_grid_shape())
#     center = np.array([0, 0, fmap._zmin + fmap._zrange])
#     scale = 2.0 * mesh_scale / max(fmap._xrange, fmap._yrange, fmap._zrange)
#     vertices = (vertices - center) * scale

#     mesh.vertices = vertices
#     sdf = mesh_to_voxels(mesh, size)
#     sdf = sdf / scale

#     if zero_fill:  # fill zeros to the inside of the object
#         return np.where(sdf >= 0, sdf, 0)
#     else:
#         return sdf


def compute_sdf(mesh, state, fmap, zero_fill=False):
    mesh_scale = 0.8
    p, q = state
    r = R.from_quat(q)
    vertices = r.apply(mesh.vertices) + p

    size = max(fmap.get_grid_shape())
    center = np.array([0, 0, fmap._zmin + fmap._zrange / 2.0])
    scale = 2.0 * mesh_scale / max(fmap._xrange, fmap._yrange, fmap._zrange)
    vertices = (vertices - center) * scale

    mesh.vertices = vertices
    sdf = mesh_to_voxels(mesh, size)
    sdf = sdf / scale

    if zero_fill:  # fill zeros to the inside of the object
        return np.where(sdf >= 0, sdf, 0)
    else:
        return sdf


def test(bs, fmap):
    def transform_mesh(objname_and_state):
        objname, state = objname_and_state
        mesh = trimesh.load(get_obj_file(objname)[0], force="mesh")
        p, q = state
        r = R.from_quat(q)
        vertices = r.apply(mesh.vertices) + p
        mesh.vertices = vertices
        return mesh

    transformed_meshes = [transform_mesh(os) for os in bs]

    start_t = time.time()
    print(f"merging meshes [{len(transformed_meshes) + 1} meshes]: ", end="", flush=True)
    scene_mesh = trimesh.util.concatenate(transformed_meshes)
    print(f"# of vertices = {len(scene_mesh.vertices)}")

    vertices = scene_mesh.vertices
    center = np.array([0, 0, fmap._zmin + fmap._zrange])
    scale = 2.0 / max(fmap._xrange, fmap._yrange, fmap._zrange)
    vertices = (vertices - center) * scale
    scene_mesh.vertices = vertices
    message(f"{time.time() - start_t:.2f}[sec]")

    start_t = time.time()
    print(f"computing SDF: ", end="", flush=True)
    sdf = mesh_to_voxels(scene_mesh, 80)
    message(f"{time.time() - start_t:.2f}[sec]")

    return sdf


def normal_distribution(fmap, contact_position):
    mean = contact_position
    sigma = fmap.bandwidth
    v = [np.exp(-(((np.linalg.norm(p - mean)) / sigma) ** 2)) for p in fmap.positions]
    v = np.array(v).reshape(fmap.get_grid_shape())
    return v


def get_obj_file(object_name):
    if object_name == "table":
        p = Path(object_info._object_dir)
        obj_file = str(p / "env" / "table_surface.obj")
        # mesh_scale = 1.0
        scale = np.array([0.38, 0.38, 1.0])
    elif object_name == "basket":
        p = Path(object_info._object_dir)
        obj_file = str(p / "env" / "seria_basket2.obj")
        scale = np.array([1.0, 1.0, 1.0])
    else:
        obj_file, scale = object_info.obj_file(object_name)
    return obj_file, scale


def sdfs_for_objects(bin_state):
    _, _, height = fmap.get_grid_shape()
    sdfs = {}

    def sdf_for_object(object_name, object_pose):
        obj_file, scale = get_obj_file(object_name)
        mesh = trimesh.load(obj_file, force="mesh")
        mesh.vertices = scale * mesh.vertices
        print(f"{object_name}")
        sdf = compute_sdf(mesh, object_pose, fmap=fmap)
        sdf = sdf[:, :, :height]
        return sdf

    # generate SDF for the environment
    if fmap.get_scene() == "small_table":
        object_name = "table"
        sdfs[object_name] = sdf_for_object(object_name, ([0, 0, 0.68], [0, 0, 0, 1]))
    if fmap.get_scene() == "seria_basket":
        object_name = "basket"
        sdfs[object_name] = sdf_for_object(object_name, ([0, 0, 0.73], [0, 0, 0.70711, 0.70711]))

    for object_name, object_pose in bin_state:
        sdfs[object_name] = sdf_for_object(object_name, object_pose)
    return sdfs


def unsigned_sdf_for_scene(unsigned_sdfs):
    return functools.reduce(lambda x, y: np.minimum(x, y), sdfs.values())


def sdf_for_scene(sdfs):
    return functools.reduce(lambda x, y: np.minimum(x, y), sdfs.values())


def sdf_for_inside_objects(sdfs):
    nsdf = [np.where(sdf <= 0, sdf, 0) for sdf in sdfs.values()]
    return functools.reduce(lambda x, y: np.minimum(x, y), nsdf)


def compute_density(bin_state, contact_state, sigma_d=0.01):
    start_t = time.time()
    print(f"computing SDFs [{len(bin_state) + 1} SDFs]: ", end="", flush=True)
    sdfs = sdfs_for_objects(bin_state)
    message(f"{time.time() - start_t:.2f}[sec]")

    fdists = []
    weighted_fdists = []

    def get_sdf(object_name):
        return sdfs[object_name]

    start_t = time.time()
    print(f"processing contact points [{len(contact_state[0])} contacts]: ", end="", flush=True)
    for contact_position, force_value, contact_pair, contact_normal in zip(*contact_state):
        objectA, objectB = contact_pair
        # if objectA == "table" or objectB == "table":
        #     continue

        try:
            sdf1 = get_sdf(objectA)
            esdf1 = np.exp(-np.abs(sdf1) / (sigma_d))
            sdf2 = get_sdf(objectB)
            esdf2 = np.exp(-np.abs(sdf2) / (sigma_d))

            g = normal_distribution(fmap, contact_position)
            fdist = force_value * g
            fdists.append(fdist)

            unnormalized_wkde = esdf1 * esdf2 * g
            sumg = np.sum(g)  #####
            alpha = 1.0 / max(np.sum(unnormalized_wkde), 1e-4) * sumg  #####
            normalized_wkde = alpha * force_value * unnormalized_wkde
            weighted_fdists.append(normalized_wkde)
        except:
            message("skip a contact point (contacting object is out of bound)")

    message(f"{time.time() - start_t:.2f}[sec]")

    # start_t = time.time()
    force_distribution = functools.reduce(operator.add, fdists)
    weighted_force_distribution = functools.reduce(operator.add, weighted_fdists)
    scene_sdf = sdf_for_scene(sdfs)
    # print("fdist sum = ", np.sum(force_distribution), "wfdist sum =", np.sum(weighted_force_distribution))
    # message(f"post process: {time.time() - start_t:.2f}[sec]")

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
        bin_state, contacts = load(frameNo)
        d = compute_density(bin_state, contacts)
        if log_scale:
            d = np.log(1 + d)
        pd.to_pickle(d, out_file)


def compute_force_distribution_for_all(scene_numbers=range(0, 10)):
    start_tm = time.time()
    with futures.ProcessPoolExecutor() as executor:
        executor.map(compute_force_distribution, scene_numbers)
    print(f"compute force distribution took: {time.time() - start_tm} [sec]")


def visualize(bin_state, d, scale=1.0, draw_range=[0.3, 0.9]):
    fmap.set_values(d * scale)
    viewer.publish_bin_state(bin_state, fmap, draw_fmap=True, draw_range=draw_range)


# compute_force_distribution_for_all()


if __name__ == "__main__":
    bin_state, contact_state = load(81)
    d = compute_density(bin_state, contact_state)
    bs_v = [b for b in bin_state if b[0] != "table"]
    visualize(bs_v, d[0], scale=1e1, draw_range=[0.03, 0.9])
