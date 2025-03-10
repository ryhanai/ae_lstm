import functools
import operator
import os
import time
from pathlib import Path

from fmdev import forcemap
# import mesh2sdf
import numpy as np
import pandas as pd
import trimesh
from core.object_loader import ObjectInfo
from core.utils import message

from mesh_to_sdf import mesh_to_voxels
from scipy.spatial.transform import Rotation as R

## config for T-RO '24, piling on the table, train with YCB+Conveni-v1
# data_dir = f"{os.environ['HOME']}/Dataset/forcemap/tabletop240304"
# object_info = ObjectInfo("ycb_conveni_v1")
# fmap = forcemap.GridForceMap("small_table", bandwidth=0.03)
# def in_forcemap_area(p):
#     return p[0] > -0.25 and p[0] < 0.25 and p[1] > -0.25 and p[1] < 0.25 and p[2] > 0.685 and p[2] < 1.05


## config for the latest basket scene
# data_dir = f"{os.environ['HOME']}/Dataset/forcemap/basket240511"
# object_info = ObjectInfo("ycb_conveni_v1_small")


class FmapSmoother:
    def __init__(self, env_config, viewer=None):
        self._data_dir = env_config['data_dir']
        self._object_info = ObjectInfo(env_config['object_set'])
        self._fmap = forcemap.GridForceMap(env_config['forcemap'], 
                                           bandwidth=env_config['forcemap_bandwidth'])
        if viewer != None:
            self._viewer = viewer
            self._viewer.set_object_info(self._object_info)

        self._voxel_resolution = 40

        if env_config['use_precomputed_sdfs']:
            sdf_file = 'sdfs.pkl'
            message(f'load precomputed SDFs from {sdf_file}')
            self._sdfs = pd.read_pickle('sdfs.pkl')
        else:
            self._sdfs = {}

    def load(self, scene_number, exclude_protruding_object=False):
        bin_state = pd.read_pickle(f"{self._data_dir}/bin_state{scene_number:05d}.pkl")
        contact_state = pd.read_pickle(f"{self._data_dir}/contact_raw_data{scene_number:05d}.pkl")
        if exclude_protruding_object:
            bin_state = [s for s in bin_state if in_forcemap_area(s[1][0])]
        return bin_state, contact_state

    # def compute_sdf(self, mesh, state, zero_fill=False):
    #     mesh_scale = 0.8
    #     p, q = state
    #     r = R.from_quat(q)
    #     vertices = r.apply(mesh.vertices) + p

    #     size = max(self._fmap.get_grid_shape())
    #     center = np.array([0, 0, self._fmap._zmin + self._fmap._zrange / 2.0])
    #     scale = 2.0 * mesh_scale / max(self._fmap._xrange, self._fmap._yrange, self._fmap._zrange)
    #     vertices = (vertices - center) * scale

    #     mesh.vertices = vertices
    #     # sdf = mesh_to_voxels(mesh, size)
    #     sdf = self.mesh_to_voxels(mesh, size)
    #     sdf = sdf / scale

    #     if zero_fill:  # fill zeros to the inside of the object
    #         return np.where(sdf >= 0, sdf, 0)
    #     else:
    #         return sdf

    def transform_forcemap_grids(self, object_name, object_pose):
        """
            transform forcemap grid coords into coordinates in SDF coords
        """
        query_points = self._fmap.positions
        sdf, orig_center, scale = self.get_sdf_for_object(object_name)
        p, q = object_pose
        rot = R.from_quat(q)
        p_in_sdf_coords = (rot.inv().apply(query_points - p) - orig_center) * scale
        return p_in_sdf_coords

    def get_sdf_values(self, sdf, query_points):
        r = self._voxel_resolution
        indices = np.round((query_points - (-1)) / (2 / self._voxel_resolution)).astype(np.int32)

        def f(x):
            return sdf[x[0]-1, x[1]-1, x[2]-1] if 0<=x[0]<=r and 0<=x[1]<=r and 0<=x[2]<=r else 10.

        return indices, np.apply_along_axis(f, 1, indices)
    
    def get_sdf_for_object_in_scene(self, object_name, object_pose):
        ps = self.transform_forcemap_grids(object_name, object_pose)
        sdf, orig_center, scale = self.get_sdf_for_object(object_name)
        indices, values = self.get_sdf_values(sdf, ps)
        return values.reshape(self._fmap.get_grid_shape())

    def compute_sdf_for_object(self, object_name):
        print(f"computing SDF for {object_name}: ", end="", flush=True)
        start_t = time.time()

        obj_file, mesh_unit_scale = self.get_obj_file(object_name)
        mesh = trimesh.load(obj_file, force="mesh")
        mesh.vertices = mesh_unit_scale * mesh.vertices

        scale = 2 / np.max(mesh.extents)
        sdf = mesh_to_voxels(mesh, voxel_resolution=self._voxel_resolution)
        sdf = sdf / scale
        orig_center = mesh.bounding_box.centroid

        message(f"{time.time() - start_t:.2f}[sec]")        
        return sdf, orig_center, scale

    def precompute_sdfs_for_all_objects(self, save_sdfs=True):
        for object_name in self._object_info.names():
            self.get_sdf_for_object(object_name)
        if save_sdfs:
            pd.to_pickle(self._sdfs, 'sdfs.pkl')

    def get_sdf_for_object(self, object_name):
        """
            compute SDF on demand
        """
        sdf = self._sdfs.get(object_name, None)
        if sdf == None:            
            sdf = self.compute_sdf_for_object(object_name)
            self._sdfs[object_name] = sdf
        return sdf            

    def compute_density(self, bin_state, contact_state, sigma_d=0.01):
        bs = dict(bin_state)
        assert self._fmap.get_scene() == "small_table" or self._fmap.get_scene() == "seria_basket"
        if self._fmap.get_scene() == "small_table":
            bs['table'] = ([0, 0, 0.68], [0, 0, 0, 1])
        if self._fmap.get_scene() == "seria_basket":
            bs['seria_basket'] = ([0, 0, 0.73], [0, 0, 0.70711, 0.70711])

        fdists = []
        weighted_fdists = []

        start_t = time.time()
        print(f"processing contact points [{len(contact_state[0])} contacts]: ", end="", flush=True)
        for contact_position, force_value, contact_pair, contact_normal in zip(*contact_state):
        # for contact_position, force_value, contact_pair in zip(*contact_state):            
            objectA, objectB = contact_pair

            sdf1 = self.get_sdf_for_object_in_scene(objectA, bs[objectA])
            esdf1 = np.exp(-np.abs(sdf1) / (sigma_d))
            sdf2 = self.get_sdf_for_object_in_scene(objectB, bs[objectB])
            esdf2 = np.exp(-np.abs(sdf2) / (sigma_d))

            g = self.normal_distribution(contact_position)
            fdist = force_value * g
            fdists.append(fdist)

            unnormalized_wkde = esdf1 * esdf2 * g
            sumg = np.sum(g)  #####
            alpha = 1.0 / max(np.sum(unnormalized_wkde), 1e-4) * sumg  #####
            normalized_wkde = alpha * force_value * unnormalized_wkde
            weighted_fdists.append(normalized_wkde)
            # message("skip a contact point (contacting object is out of bound)")
                
        force_distribution = functools.reduce(operator.add, fdists)
        weighted_force_distribution = functools.reduce(operator.add, weighted_fdists)

        # sdf for the scene
        inside_only = False
        sdfs = [self.get_sdf_for_object_in_scene(name, pose) for name, pose in bs.items()]

        if inside_only:  # nsdf
            sdfs = [np.where(sdf <= 0, sdf, 0) for sdf in sdfs]
        scene_sdf = functools.reduce(lambda x, y: np.minimum(x, y), sdfs)

        message(f"{time.time() - start_t:.2f}[sec]")

        return force_distribution, weighted_force_distribution, scene_sdf

    def normal_distribution(self, contact_position):
        mean = contact_position
        sigma = self._fmap.bandwidth
        v = [np.exp(-(((np.linalg.norm(p - mean)) / sigma) ** 2)) for p in self._fmap.positions]
        v = np.array(v).reshape(self._fmap.get_grid_shape())
        return v

    def get_obj_file(self, object_name):
        if object_name == "table" or object_name == "simple_table":
            p = Path(self._object_info._object_dir)
            obj_file = str(p / "env" / "table_surface.obj")
            # mesh_scale = 1.0
            scale = np.array([0.38, 0.38, 1.0])
        elif object_name == "seria_basket":
            p = Path(self._object_info._object_dir)
            obj_file = str(p / "env" / "seria_basket2.obj")
            scale = np.array([1.0, 1.0, 1.0])
        else:
            obj_file, scale = self._object_info.obj_file(object_name)
        return obj_file, scale

    def visualize(self, bin_state, d, scale=1.0, draw_range=[0.3, 0.9]):
        self._fmap.set_values(d * scale)
        self._viewer.publish_bin_state(bin_state, self._fmap, draw_fmap=True, draw_range=draw_range)

    def compute_force_distribution(self, frameNo, log_scale=False, overwrite=False):
        out_file = os.path.join(self._data_dir, "force_zip{:05d}.pkl".format(frameNo))
        if (not overwrite) and os.path.exists(out_file):
            print(f"skip [{frameNo}]")
        else:
            print(f"process [{frameNo}], log_scale={log_scale}")
            bin_state, contacts = self.load(frameNo)
            d = self.compute_density(bin_state, contacts)
            if log_scale:
                d = np.log(1 + d)
            pd.to_pickle(d, out_file)


def in_forcemap_area(p):
    return p[0] > -0.18 and p[0] < 0.18 and p[1] > -0.18 and p[1] < 0.18 and p[2] > 0.70 and p[2] < 1.05


def in_forcemap_area_mesh2sdf(p):
    """need to remove objects outside of the forcemap strictly in case of mesh2sdf"""
    return p[0] > -0.2 and p[0] < 0.2 and p[1] > -0.2 and p[1] < 0.2 and p[2] > 0.735 and p[2] < 1.0


# def get_obj_position(bin_state, object_name):
#     return [o[1] for o in bin_state if o[0] == object_name][0]
#     # try:
#     #     return [o[1] for o in bin_state if o[0] == object_name][0]
#     # except IndexError:
#     #     return (np.array([0, 0, 0.73]), R.from_euler("xyz", [0, 0, 1.57080], degrees=False).as_quat())


# def compute_sdf_with_mesh2sdf(mesh, state, fmap, zero_fill=False):
#     # mesh_scale = 0.8
#     mesh_scale = 1.0
#     p, q = state
#     r = R.from_quat(q)
#     # r.as_matrix()
#     vertices = r.apply(mesh.vertices) + p

#     size = max(fmap.get_grid_shape())
#     center = np.array([0, 0, fmap._zmin + fmap._zrange])
#     scale = 2.0 * mesh_scale / max(fmap._xrange, fmap._yrange, fmap._zrange)
#     vertices = (vertices - center) * scale

#     sdf = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=2 / size, return_mesh=False)
#     sdf = sdf / scale

#     if zero_fill:  # fill zeros to the inside of the object
#         return np.where(sdf >= 0, sdf, 0)
#     else:
#         return sdf


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


if __name__ == "__main__":
    bin_state, contact_state = fmap_smoother.load(81)
    d = fmap_smoother.compute_density(bin_state, contact_state)
    bs_v = [b for b in bin_state if b[0] != "table"]
    fmap_smoother.visualize(bs_v, d[0], scale=1e1, draw_range=[0.03, 0.9])

    # fmap_smoother.compute_force_distribution_for_all()