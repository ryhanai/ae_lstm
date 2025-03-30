from typing import List

import numpy as np
from omni.isaac.core.prims.xform_prim import XFormPrim
from aist_sb_ur5e.factory import create_convex_rigid_body


class TabletopEnv:
    """
    Environment for picking on a table
    """

    def __init__(
        self,
        env_usd_path: str = "aist_sb_ur5e/static/usd/green_table_scene.usd",
    ) -> None:
        """
        Initialize TabletopEnv

        Args:
            store_usd_path (str, optional):
                USD file path of the environment
                Defaults to "......".
        """

        self._shelf: XFormPrim = create_convex_rigid_body(
            name="table",
            prim_path="/World/table_surface",
            usd_path=env_usd_path,
            mass=100.0,
            static_friction=0.4,
            dynamic_friction=0.3,
            position=np.array([0.0, 0.0, 0.71]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            kinematic=True,
        )

        # self._products: List = self.load_scene()

    # @property
    # def products(self) -> List[XFormPrim]:
    #     """
    #     product list

    #     Returns:
    #         List[XFormPrim]: product list
    #     """
    #     return self._products

    # @property
    # def shelf(self) -> XFormPrim:
    #     """
    #     棚

    #     Returns:
    #         XFormPrim: 棚
    #     """
    #     return self._shelf

