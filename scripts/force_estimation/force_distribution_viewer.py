# -*- coding: utf-8 -*-

import colorsys

import numpy as np
import scipy.linalg
from core.object_loader import ObjectInfo
from force_estimation import forcemap, rviz_client

object_info = ObjectInfo("ycb_conveni_v1")


class ForceDistributionViewer:
    """
    Singleton pattern
    Duplicated instantiation causes the error of ROS node intialization
    """

    _unique_instance = None

    def __new__(cls):
        raise NotImplementationError("Cannot generate instance by constructor")

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_instance(cls):
        if not cls._unique_instance:
            cls._unique_instance = cls.__internal_new__()
            cls.rviz_client = rviz_client.RVizClient()
        return cls._unique_instance

    def publish_bin_state(self, bin_state, fmap, draw_fmap=True, draw_force_gradient=False, draw_range=[0.5, 0.9]):
        self.rviz_client.delete_all()
        self.draw_bin(fmap)

        positions = fmap.get_positions()
        fvals = fmap.get_values()

        if bin_state is not None:
            self.draw_objects(bin_state, fmap)
        if draw_fmap:
            self.draw_force_distribution(positions, fvals, draw_range=draw_range)
        if draw_force_gradient:
            self.draw_force_gradient(positions, fvals)
        self.rviz_client.show()

    def draw_bin(self, fmap):
        scene = fmap.get_scene()
        if scene == "seria_basket" or scene == "small_table":
            mesh_file = "seria_basket.dae"
            mesh_pose = ([0, 0, 0.73], [0, 0, 0.70711, 0.70711])
            scale = [1, 1, 1]
        elif scene == "konbini_shelf":
            mesh_file = "simple_shelf.obj"
            mesh_pose = ([0, 0, 0], [0, 0, 0, 1])
            scale = [0.01, 0.01, 0.01]
        else:
            print(f"[VIEWER] unknown scene: {scene}")
            return

        self.rviz_client.draw_mesh(
            f"package://force_estimation/meshes_extra/{mesh_file}", mesh_pose, rgba=(0.5, 0.5, 0.5, 0.2), scale=scale
        )

    def draw_objects(self, bin_state, fmap):
        for object_state in bin_state:
            name, pose = object_state

            mesh_file, scale = object_info.rviz_mesh_file(name)
            assert mesh_file, f"mesh file for {name} not found"

            self.rviz_client.draw_mesh(
                mesh_file,
                pose,
                (0.5, 0.5, 0.5, 0.3),
            )

    def draw_force_distribution(self, positions, fvals, draw_range=[0.5, 0.9]):
        fvals = fvals.flatten()
        fmax = np.max(fvals)
        fmin = np.min(fvals)
        points = []
        rgbas = []
        if fmax - fmin < 1e-3:
            print("the range of force values too small")
            return
        # std_fvals = (fvals - fmin) / (fmax - fmin)

        for (x, y, z), f in zip(positions, fvals):
            if draw_range[0] <= f and f <= draw_range[1]:
                points.append([x, y, z])
                hue = max(0, (0.7 - f) / 0.7)
                r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
                rgbas.append([r, g, b, 1])
        self.rviz_client.draw_points(points, rgbas)

    def draw_vector_field(self, positions, values, scale=0.5):
        self.rviz_client.draw_arrows(positions, positions + values * scale)

    def draw_force_gradient(self, positions, fvals, scale=0.3, threshold=0.008):
        gxyz = np.gradient(-fvals)
        g_vecs = np.column_stack([g.flatten() for g in gxyz])
        pos_val_pairs = [(p, g) for (p, g) in zip(positions, g_vecs) if scipy.linalg.norm(g) > threshold]
        positions, values = zip(*pos_val_pairs)
        self.draw_vector_field(np.array(positions), np.array(values), scale=scale)
