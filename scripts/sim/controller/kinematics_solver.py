import os
from typing import Optional
from pathlib import Path

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver


HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()


class KinematicsSolver(ArticulationKinematicsSolver):
    """Kinematics Solver for UR5e robot.  This class loads a LulaKinematicsSolver object

    Args:
        robot_articulation (SingleArticulation): An initialized Articulation object representing this UR5e
        end_effector_frame_name (Optional[str]): The name of the UR5e end effector.  If None, an end effector link will
            be automatically selected.  Defaults to None.
        attach_gripper (Optional[bool]): If True, a URDF will be loaded that includes a suction gripper.  Defaults to False.
    """

    def __init__(
        self,
        robot_articulation: SingleArticulation,
        end_effector_frame_name: Optional[str] = None,
        attach_gripper: Optional[bool] = False,
    ) -> None:

        # mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        # robot_urdf_path = os.path.join(mg_extension_path, HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/urdf/ur5e.urdf")
        # robot_description_yaml_path = os.path.join(
        #     mg_extension_path, HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/robot_descriptor.yml"
        # )

        # self._kinematics = LulaKinematicsSolver(
        #     robot_description_path=robot_description_yaml_path, urdf_path=robot_urdf_path
        # )

        self._kinematics = robot_articulation._kinematics

        if end_effector_frame_name is None:
            end_effector_frame_name = "tool0"

        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)

        return
