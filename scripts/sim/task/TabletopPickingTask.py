from typing import Any

import numpy as np
import pandas as pd
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.sensors.camera import Camera
from omni.kit.viewport.window.window import ViewportWindow
from pxr.Usd import Prim

from aist_sb_ur5e.env import get_settings
from aist_sb_ur5e.model import RMPFlowTarget, UR5e
from aist_sb_ur5e.model.factory import create_camera, create_light, create_viewport


class TabletopPickingTask(BaseTask):
    """
    Task for picking on a table

    Args:
        BaseTask ([type]): [description]
    """

    def __init__(self, static_path: str = "aist_sb_ur5e/static") -> None:
        """
        Initialize ConveniPickupTask

        Args:
            static_path (str, optional):
                静的ファイルを格納したディレクトリのパス.
                Defaults to "aist_sb_ur5e/static".
        """
        super().__init__(
            name="conveni_pickup_task",
        )
        self._ur5e = UR5e(
            usd_path=f"{static_path}/usd/ur5e_with_gripper_and_frame_410.usd",
        )
        self._rmpflow_target: XFormPrim = RMPFlowTarget()
        self._convencience_store = ConvenienceStore(
            # store_usd_path=f"{static_path}/usd/table_surface.usd",
            store_usd_path=f"{static_path}/usd/tana_zentai.usd",            
            goods_usd_path=f"{static_path}/usd/java_curry_chukara/java_curry_chukara.usd",
            gap=get_settings().conveni_product_stack_gap,
            num_product_rows=get_settings().conveni_product_stack_rows,
            num_product_columns=get_settings().conveni_product_stack_columns,
        )
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
        for sensor in self._ur5e.sensors:
            scene.add(obj=sensor)
        scene.add(obj=self._rmpflow_target)
        scene.add(obj=self._convencience_store.shelf)
        for product in self._convencience_store.products:
            scene.add(obj=product)
        scene.add_default_ground_plane()
        self.post_reset()
        return

    def post_reset(self) -> None:
        """
        ロボットを初期状態にリセットする
        """
        self._ur5e.set_joints_default_state(
            positions=np.concatenate(
                [
                    np.array(
                        [
                            -np.pi / 2,
                            -np.pi / 2,
                            -np.pi / 2,
                            -np.pi / 2,
                            np.pi / 2,
                            np.pi,
                        ]
                    ),
                    np.zeros(6),
                ]
            )
        )
        self._rmpflow_target.set_default_state(
            position=np.array([0.13, 0.49, 1.52]),
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
                    self._convencience_store.shelf.name,
                    *[curry.name for curry in self._convencience_store.products],
                ],
                "modifiable": False,
            },
            "contact_sensor_names": {
                "value": [sensor.name for sensor in self._ur5e.sensors],
                "modifiable": False,
            },
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
        bs = pd.read_pickle(f'/home/artuser/Dataset/forcemap/tabletop240304/bin_state{scene_idx:05d}.pkl')
        print(bs)
        for name, (p, o) in bs:
            for product in self._convencience_store.products:
                if product.name == name:
                    # print(dir(p))
                    o[0], o[1], o[2], o[3] = o[3], o[0], o[1], o[2]
                    print(f'SET POSE: {name}')
                    product.set_world_pose(p, o)
                    self._active_products.append(product)

    def get_active_products(self):
        return self._active_products
    