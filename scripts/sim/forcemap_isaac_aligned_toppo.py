# -*- coding: utf-8 -*-

import os
import numpy as np
from sim.forcemap_isaac import RandomTableScene, DatasetGenerator
from omni.isaac.core import World


env_config = {
    'headless': False,
    'asset_root': os.environ["HOME"] + "/Dataset/scenes",
    'scene_usd': "green_table_scene.usd",
    'viewpoint': ([-0.42, 0, 1.15], [0, 45, 0]),  # euler angles in degree
    'viewpoint_randomization_range': (0.02, 3),  # ([m], [degree])
    'object_set': "ycb_conveni_v1",
    'resize_method': 'crop',  # 'crop' | 'resize'
    'image_resolution': [640, 480],
}


class AlignedToppoScene(Scene):
    def __init__(self):
        self.create_toppos(20)

    def create_toppos(self, n):
        self._loaded_objects = []
        usd_file = os.path.join(
            os.environ["HOME"],
            "Dataset/Konbini/VER002/Seamless/vt2048/15_TOPPO-ART-vt2048SA/15_TOPPO-ART-pl2048SA/15_TOPPO-ART-pl2048SA.usd",
        )
        for i in range(n):
            self._loaded_objects.append(Object(usd_file, i, self._world))

    def create_toppo_scene(self, n):
        global used_objects
        used_objects = []

        self._world.reset()
        for i, o in enumerate(self._loaded_objects):
            prim = o.get_primitive()
            pos_xy = np.array([0.0, -0.1]) + np.array([0.0, 0.025]) * i
            pos_z = 0.73

            axis = [1, 0, 0]
            angle = 90
            set_pose(prim, ([pos_xy[0], pos_xy[1], pos_z], (axis, angle)))
            used_objects.append(o)