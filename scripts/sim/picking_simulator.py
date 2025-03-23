from operator import itemgetter

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper

from aist_sb_ur5e.controller import RMPFlowController, SpaceMouseController
from aist_sb_ur5e.task import ConveniPickupTask


my_world: World = World(stage_units_in_meters=1.0)

conveni_task = ConveniPickupTask(static_path="/home/ryo/Program/hello-isaac-sim/aist_sb_ur5e/static")
my_world.add_task(task=conveni_task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(
    name=conveni_task.get_params()["robot_names"]["value"][0]
)
target: XFormPrim = my_world.scene.get_object(
    name=conveni_task.get_params()["x_form_prim_names"]["value"][0]
)
gripper: ParallelGripper = conveni_task.get_params()["gripper"]["value"]

rmpflow_controller = RMPFlowController(robot_articulation=ur5e,
                                       robot_description_path='/home/ryo/Program/hello-isaac-sim/aist_sb_ur5e/static/rmpflow/robot_descriptor.yml',
                                       rmpflow_config_path='/home/ryo/Program/hello-isaac-sim/aist_sb_ur5e/static/rmpflow/ur5e_rmpflow_common.yml',
                                       urdf_path='/home/ryo/Program/hello-isaac-sim/aist_sb_ur5e/static/urdf/ur5e.urdf',
)
target_controller = SpaceMouseController(device_type="SpaceMouse Compact", rotate_gain=0.3)

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
            gripper.apply_action(
                ArticulationAction(
                    joint_positions=itemgetter(7, 9)(gripper_action.joint_positions)
                )
            )

simulation_app.close()
