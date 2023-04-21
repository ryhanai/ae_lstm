from omni.isaac.kit import SimulationApp
import os

CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 960, "height": 540, "num_frames": 5}
simulation_app = SimulationApp(launch_config=CONFIG)

ENV_URL = "/home/ryo/Downloads/Collected_ycb_piled_scene/simple_shelf_scene.usd"
OBJECTS_URL_SHOP = "/home/ryo/Dataset/Konbini/VER002/Seamless/vt2048"
SCOPE_NAME = "/MyScope"

import carb
import random
import math
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage, open_stage

import omni.replicator.core as rep

rep.settings.carb_settings("/omni/replicator/RTSubframes", 2)

world = World(stage_units_in_meters=1.0)

usd_files = rep.utils.get_usd_files(OBJECTS_URL_SHOP, recursive=True)

def place_objects():
    usd_files = rep.utils.get_usd_files(OBJECTS_URL_SHOP, recursive=True)
    objects = rep.randomizer.instantiate(usd_files,
                                            size=20,
                                            with_replacements=True,
                                            mode='scene_instance',
                                            use_cache=True)

    with objects:
        rep.modify.pose(position=rep.distribution.uniform((-0.1, -0.1, 0.8), (0.1, 0.1, 0.9)))
        rep.physics.rigid_body(velocity=rep.distribution.uniform((0, 0, -0.01), (0, 0, 0)),
                                angular_velocity=rep.distribution.uniform((0, 0, 0), (0, 0, 0)))

    return objects.node

open_stage(ENV_URL)
rep.randomizer.register(place_objects)
cam = rep.create.camera(clipping_range=(0.1, 5.0))

# with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
with rep.trigger.on_time(interval=1, num=CONFIG["num_frames"]):
    rep.randomizer.place_objects()
    # rep.randomizer.randomize_lights()

    with cam:
        rep.modify.pose(
            position=rep.distribution.uniform((0.56, -0.02, 1.3), (0.6, 0.02, 1.35)),
            rotation=rep.distribution.uniform((-2, -38, -2), (2, -34, 2)),
        )

# Starts replicator and waits until all data was successfully written
def run_orchestrator():
    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        print(1)
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        print(2)
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


run_orchestrator()
