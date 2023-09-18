import numpy as np
import pandas as pd
import trimesh
import mesh2sdf
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import forcemap
from force_estimation import force_distribution_viewer
import functools
import operator
from concurrent import futures
import os
import time


fmap = forcemap.GridForceMap('seria_basket', bandwidth=0.03)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


data_dir = f"{os.environ['HOME']}/Dataset/dataset2/basket-filling230829"
obj_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/specification/meshes/objects"

def compute_df(mesh, state, size, scale):
    p, q = state
    r = R.from_quat(q)
    # r.as_matrix()

    vertices = r.apply(mesh.vertices) + p - np.array([0, 0, 0.73+0.13])
    vertices = vertices * scale

    sdf = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=2/size, return_mesh=False)
    # m.export('hoge.obj')
    df = np.where(sdf >= 0, sdf, 0)('seria_basket', bandwidth=0.03)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


data_dir = f"{os.environ['HOME']}/Dataset/dataset2/basket-filling230829"
obj_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/specification/meshes/objects"

def compute_df(mesh, state, size, scale):
    p, q = state
    r = R.from_quat(q)
    # r.as_matrix()

    vertices = r.apply(mesh.vertices) + p - np.array([0, 0, 0.73+0.13])
    vertices = vertices * scale

    sdf = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=2/size, return_mesh=False)
    # m.export('hoge.obj')
    df = np.where(sdf >= 0, sdf, 0)
    return df


def normal(fmap, contact_position):
    mean = contact_position
    sigma = fmap.bandwidth
    v = [np.exp(-((np.linalg.norm(p-mean))/sigma)**2) for p in fmap.positions]
    v = np.array(v).reshape((40, 40, 40))
    return v


def get_obj_file(name):
    if name == 'seria_basket':
        return f'{obj_dir}/seria_basket_body_collision.obj', 0.001
    else:
        return f'{obj_dir}/ycb/{name}/google_16k/textured.obj', 1.0


def get_obj_position(bin_state, object_name):
    try:
    return df


def normal(fmap, contact_position):
    mean = contact_position
    sigma = fmap.bandwidth
    v = [np.exp(-((np.linalg.norm(p-mean))/sigma)**2) for p in fmap.positions]
    v = np.array(v).reshape((40, 40, 40))
    return v


def get_obj_file(name):
    if name == 'seria_basket':
        return f'{obj_dir}/seria_basket_body_collision.obj', 0.001
    else:
        return f'{obj_dir}/ycb/{name}/google_16k/textured.obj', 1.0


def get_obj_position(bin_state, object_name):
    try:
        return [o[1] for o in bin_state if o[0] == object_name][0]
    except IndexError:
        return (np.array([0, 0, 0.73]), R.from_euler('xyz', [0,0,1.57080], degrees=False).as_quat())


def compute_density(scene_number=38,
                    size=40,
                    scale=1/0.13,
                    sigma_d=0.01):
    bin_state = pd.read_pickle(f'{data_dir}/bin_state{scene_number:05}.pkl')
    contacts = pd.read_pickle(f'{data_dir}/contact_raw_data{scene_number:05}.pkl')

    kde_dists = []
    weighted_dists = []

    for contact_position, force_value, contact_pair in zip(*contacts):
        objectA, objectB = contact_pair
        print(objectA, objectB)
        obj_file1, mesh_scale1 = get_obj_file(objectA)
        obj_file2, mesh_scale2 = get_obj_file(objectB)
        mesh1 = trimesh.load(obj_file1, force='mesh')
        mesh1.vertices = mesh_scale1 * mesh1.vertices
        mesh2 = trimesh.load(obj_file2, force='mesh')
        mesh2.vertices = mesh_scale2 * mesh2.vertices

        g = normal(fmap, contact_position)
        s1 = get_obj_position(bin_state, objectA)
        s2 = get_obj_position(bin_state, objectB)

        df1 = compute_df(mesh1, s1, size=size, scale=scale)
        edf1 = np.exp(-df1/(sigma_d*scale))
        df2 = compute_df(mesh2, s2, size=size, scale=scale)
        edf2 = np.exp(-df2/(sigma_d*scale))

        # return edf1, edf2, g
        # kde_dists.append(g)

        unnormalized_wkde = edf1 * edf2 * g
        alpha = 1. / max(np.sum(unnormalized_wkde), 1e-3)
        normalized_wkde = alpha * force_value * unnormalized_wkde
        weighted_dists.append(normalized_wkde)

    # return bin_state, functools.reduce(operator.add, kde_dists), functools.reduce(operator.add, weighted_dists)

    return bin_state, functools.reduce(operator.add, weighted_dists)


def compute_force_distribution(frameNo, log_scale=True, overwrite=False):
    out_file = os.path.join(data_dir, 'force_zip{:05d}.pkl'.format(frameNo))
    if (not overwrite) and os.path.exists(out_file):
        print(f'skip [{frameNo}]')
    else:
        print(f'process [{frameNo}]')
        bs, d = compute_density(frameNo)
        if log_scale:
            d = np.log(1 + d)
        pd.to_pickle(d, out_file)


def compute_force_distribution_for_all(scene_numbers=range(0, 2000)):
    start_tm = time.time()
    with futures.ProcessPoolExecutor() as executor:
        executor.map(compute_force_distribution, scene_numbers)
    print(f'compute force distribution took: {time.time() - start_tm} [sec]')


def visualize(bin_state, d, scale=0.4):
    fmap.set_values(d * scale)
    viewer.publish_bin_state(bin_state, fmap)
