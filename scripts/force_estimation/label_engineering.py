import numpy as np
import pandas as pd
import trimesh
import mesh2sdf
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import forcemap


fmap = forcemap.GridForceMap('seria_basket', bandwidth=0.04)


def compute_df(obj_file, state, size, scale):
    mesh = trimesh.load(obj_file, force='mesh')

    p, q = state[1]
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

def test(scene_number=28, size=40, scale=1/0.13, sigma_d=0.01):
    obj_file1 ='~/Downloads/ycb/009_gelatin_box/google_16k/textured.obj'
    obj_file2 ='~/Downloads/ycb/010_potted_meat_can/google_16k/textured.obj'
    s = pd.read_pickle(f'/Users/ryo/Dataset/dataset2/basket-filling3/bin_state{scene_number:05}.pkl')
    contacts = pd.read_pickle(f'/Users/ryo/Dataset/dataset2/basket-filling3/contact_raw_data{scene_number:05}.pkl')
    i = 1
    ci = contacts[0][i]
    fi = contacts[1][i]
    g = normal(fmap, ci)
    s1 = s[0]
    s2 = s[3]
    df1 = compute_df(obj_file1, s1, size=size, scale=scale)
    edf1 = np.exp(-df1/(sigma_d*scale))
    df2 = compute_df(obj_file2, s2, size=size, scale=scale)
    edf2 = np.exp(-df2/(sigma_d*scale))
    return edf1, edf2, g

