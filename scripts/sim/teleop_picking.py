from operator import itemgetter
from pathlib import Path

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
import json

import cv2
import cv_bridge
import numpy as np
import rclpy
import transforms3d as tf
from aist_sb_ur5e.controller import RMPFlowController, SpaceMouseController
from aist_sb_ur5e.model.factory import create_contact_sensor
from dataset.object_loader import ObjectInfo
from isaacsim.core.utils.transformations import pose_from_tf_matrix, tf_matrix_from_pose
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper
from sensor_msgs.msg import Image
from sim.controller.kinematics_solver import KinematicsSolver
from sim.task.tabletop_picking_task import TabletopPickingTask
from visualization_msgs.msg import Marker, MarkerArray

HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()

my_world: World = World(stage_units_in_meters=1.0)

# change gravity
# my_world.get_physics_context().set_gravity(0.0)
# my_world.set_gravity(value=-9.81 / meters_per_unit())

task = TabletopPickingTask(static_path=HELLO_ISAAC_ROOT / "aist_sb_ur5e/static")
my_world.add_task(task=task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(name=task.get_params()["robot_names"]["value"][0])
target: XFormPrim = my_world.scene.get_object(name=task.get_params()["x_form_prim_names"]["value"][0])
gripper: ParallelGripper = task.get_params()["gripper"]["value"]

rmpflow_controller = RMPFlowController(
    robot_articulation=ur5e,
    robot_description_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/robot_descriptor.yml"),
    rmpflow_config_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/ur5e_rmpflow_common.yml"),
    urdf_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/urdf/ur5e.urdf"),
)


target_controller = SpaceMouseController(device_type="SpaceMouse Compact", rotate_gain=0.3)
task.load_bin_state(scene_idx=13)
ur5e._kinematics.set_robot_base_pose(*ur5e.get_world_pose())


while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        target_position, target_orientation = target.get_world_pose()
        next_position, next_orientation = target_controller.forward(
            position=target_position,
            orientation=target_orientation,
        )
        target.set_world_pose(
            position=next_position,
            orientation=next_orientation,
        )

        action: ArticulationAction = rmpflow_controller.forward(
            target_end_effector_position=next_position,
            target_end_effector_orientation=next_orientation,
        )
        ur5e.get_articulation_controller().apply_action(control_actions=action)

        optional_action: str | None = target_controller.get_gripper_action()
        if optional_action is not None:
            gripper_action: ArticulationAction = gripper.forward(optional_action)
            gripper.apply_action(ArticulationAction(joint_positions=itemgetter(7, 9)(gripper_action.joint_positions)))

simulation_app.close()
