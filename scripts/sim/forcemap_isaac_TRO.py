# -*- coding: utf-8 -*-

import os
import numpy as np
from sim.forcemap_isaac import RandomTableScene, DatasetGenerator
from omni.isaac.core import World


env_config = {
    'headless': False,
    'asset_root': os.environ["HOME"] + "/Dataset/scenes",
    'scene_usd': "green_table_scene.usd",
    'viewpoint': ([0, 0, 1.6], [0, 90, 180]),  # euler angles in degree
    'viewpoint_randomization_range': (0.02, 3),  # ([m], [degree])
    'object_set': "ycb_conveni_v1",
}


world = World(stage_units_in_meters=1.0)
scene = RandomTableScene(world, env_config)
dataset = DatasetGenerator(scene, output_force=False)

dataset.create(2, 3)