from typing import Any

import numpy as np
import pandas as pd
from aist_sb_ur5e.env import get_settings
from aist_sb_ur5e.model import RMPFlowTarget, UR5e
from aist_sb_ur5e.model.factory import create_camera, create_light, create_viewport, create_contact_sensor
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.kit.viewport.window.window import ViewportWindow
from pxr.Usd import Prim
from sim.model.tabletop_env import TabletopEnv

import carb
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import utils, physicsUtils


class TabletopPickingTask(BaseTask):
    """
    Task for picking on a table

    Args:
        BaseTask ([type]): [description]
    """

    def __init__(self, static_path: str = "aist_sb_ur5e/static", robot: str = "ur5e") -> None:
        """
        Initialize TabletopPickingTask

        Args:
            static_path (str, optional):
                静的ファイルを格納したディレクトリのパス.
                Defaults to "aist_sb_ur5e/static".
        """
        super().__init__(
            name="tabletop_picking_task",
        )

        if robot == "ur5e":
            self._ur5e = UR5e(
                usd_path=f"{static_path}/usd/ur5e_with_gripper_and_cuboid_stand.usd",
            )
        elif robot == "franka":
            franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            franka_robot_name = "my_franka"
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._ur5e = Franka(
                prim_path=franka_prim_path,
                name=franka_robot_name,
                usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd"
            )

            self._ur5e.set_world_pose([-0.5, 0.0, 0.72], [0, 0, 0, 1])  # scalar last

            gripper_links = [
                'panda_hand',
                'panda_leftfinger',
                'panda_rightfinger',
            ]

            arm_links = [
                'panda_link0',
                'panda_link1',
                'panda_link2',
                'panda_link3',
                'panda_link4',
                'panda_link5',
                'panda_link6',
                'panda_link7',
                'panda_link8',                
            ]

            self._ur5e._sensors: list[ContactSensor] = []

            def create_contact_sensors_aux(parent_path, links, radius):
                for link in links:
                    cs = create_contact_sensor(name=link, link_path=f'{parent_path}/{link}', radius=radius)
                    self._ur5e._sensors.append(cs)

            create_contact_sensors_aux('/World/Franka', gripper_links, radius=0.2)
            create_contact_sensors_aux('/World/Franka', arm_links, radius=1.0)

            for cs in self._ur5e._sensors:
                cs.add_raw_contact_data_to_frame()

            # set friction for finger links
            phys_mat_path = "/World/Franka/phys_mat_finger"
            utils.addRigidBodyMaterial(
                stage=get_current_stage(),
                path=phys_mat_path,
                staticFriction=1.0,
                dynamicFriction=0.7,
            )
            for path in [
                "/World/Franka/panda_leftfinger",
                "/World/Franka/panda_rightfinger",
            ]:
                physicsUtils.add_physics_material_to_prim(
                    stage=get_current_stage(),
                    prim=get_current_stage().GetPrimAtPath(path),
                    materialPath=phys_mat_path,
                )


        self._rmpflow_target: XFormPrim = RMPFlowTarget()
        self._env = TabletopEnv()
        self._cameras: list[Camera] = [
            create_camera(
                name="top_camera",
                prim_path="/World/top_camera",
                position=np.array([0.00, 0.00, 1.358]),
                orientation=np.array([0, 90, 180]),
                # position=np.array([0.60, 0.65, 1.62]),
                # orientation=np.array([0, 45, -90]),
            ),
            create_camera(
                name="right_camera",
                prim_path="/World/right_camera",
                position=np.array([0.00, -0.65, 1.62]),
                # position=np.array([0.60, -0.65, 1.62]),
                orientation=np.array([0, 45, 90]),
            ),
        ]
        self._viewports: list[ViewportWindow] = [
            create_viewport(
                name="top_viewport",
                camera_path="/World/top_camera",
                position_x=1000,
                position_y=0,
            ),
            create_viewport(
                name="right_viewport",
                camera_path="/World/right_camera",
                position_x=1000,
                position_y=400,
            ),
        ]
        self._lights: list[Prim | Any] = [create_light()]
        return

    def set_up_scene(self, scene: Scene) -> None:
        """
        シーンをセットアップする

        Args:
            scene (Scene): シーン
        """
        super().set_up_scene(scene=scene)
        scene.add(obj=self._ur5e)
        for camera in self._cameras:
            scene.add(obj=camera)
        # for sensor in self._ur5e.sensors:
        #     scene.add(obj=sensor)
        scene.add(obj=self._rmpflow_target)
        for product in self._env.products:
            scene.add(obj=product)
        scene.add_default_ground_plane()
        self.post_reset()
        return

    def post_reset(self) -> None:
        """
        ロボットを初期状態にリセットする
        """
        if isinstance(self._ur5e, UR5e):
            joint_positions = np.concatenate(
                [
                    np.array(
                        [
                            0,
                            -np.pi / 2,
                            np.pi / 2,
                            -np.pi / 2,
                            -np.pi / 2,
                            np.pi,
                        ]
                    ),
                    np.zeros(6),
                ]
            )
        elif isinstance(self._ur5e, Franka):
            joint_positions = np.zeros(9)

        self._ur5e.set_joints_default_state(positions=joint_positions)
        self._rmpflow_target.set_default_state(
            position=np.array([-0.20, 0.075, 1.22]),
            orientation=euler_angles_to_quats(euler_angles=[0, np.pi, 0]),
        )

    def get_params(self) -> dict:
        """
        タスクで使用されるインスタンスの名前などを返す

        Returns:
            dict: タスクで使用されるインスタンスの名前などの辞書
        """
        return {
            "robot_names": {"value": [self._ur5e.name], "modifiable": False},
            "x_form_prim_names": {
                "value": [
                    self._rmpflow_target.name,
                    #                    self._convenience_store.shelf.name,
                    #                    *[curry.name for curry in self._convenience_store.products],
                ],
                "modifiable": False,
            },
            # "contact_sensor_names": {
            #     "value": [sensor.name for sensor in self._ur5e.sensors],
            #     "modifiable": False,
            # },
            "camera_names": {
                "value": [camera.name for camera in self._cameras],
                "modifiable": False,
            },
            "gripper": {"value": self._ur5e.gripper, "modifiable": False},
        }

    def get_observations(self) -> dict:
        """Returns current observations from the task needed for the behavioral layer at each time step.

        Observations:
                    - UR5e
                        - position
                        - velocity
                    - rmpflow_target
                        - position
                        - orientation

                Returns:
                    dict: [description]
        """
        position, orientation = self._rmpflow_target.get_world_pose()
        return {
            "UR5e": {
                "position": self._ur5e.get_joint_positions(),
                "velocity": self._ur5e.get_joint_velocities(),
            },
            "rmpflow_target": {
                "position": position,
                "orientation": orientation,
            },
        }

    def load_bin_state(self, scene_idx):
        self._active_products = []
        bs = pd.read_pickle(f"/home/ryo/Dataset/forcemap/tabletop240304/bin_state{scene_idx:05d}.pkl")
        # print(bs)
        for name, (p, o) in bs:
            for product in self._env.products:
                if product.name == name:
                    # print(dir(p))
                    o[0], o[1], o[2], o[3] = o[3], o[0], o[1], o[2]
                    # print(f"SET POSE: {name}")
                    product.set_world_pose(p, o)
                    self._active_products.append(product)

    def load_extra_large_clamp(self):
        self._active_products = []
        bs = pd.read_pickle(f"/home/ryo/Dataset/forcemap/tabletop240304/bin_state00013.pkl")
        name = "052_extra_large_clamp"
        p, o = [x for (n, x) in bs if n == name][0]
        for product in self._env.products:
            if product.name == name:
                o[0], o[1], o[2], o[3] = o[3], o[0], o[1], o[2]
                product.set_world_pose(p, o)
                self._active_products.append(product)

    def get_active_products(self):
        return self._active_products
