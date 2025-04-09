from typing import List

import numpy as np
from aist_sb_ur5e.model.factory import create_convex_rigid_body
from dataset.object_loader import ObjectInfo
from omni.isaac.core.prims.xform_prim import XFormPrim
from scipy.spatial.transform import rotation as R


class TabletopEnv:
    """
    Environment for picking on a table
    """

    def __init__(
        self,
        env_usd_path: str = "/home/ryo/Dataset/scenes/table_surface/table_surface.usd",
    ) -> None:
        """
        Initialize TabletopEnv

        Args:
            store_usd_path (str, optional):
                USD file path of the environment
                Defaults to "......".
        """

        self._table: XFormPrim = create_convex_rigid_body(
            name="table",
            prim_path="/World/table_surface",
            usd_path=env_usd_path,
            mass=100.0,
            static_friction=0.4,
            dynamic_friction=0.3,
            # position=np.array([0.0, 0.0, 0.71]),
            position=np.array([0.0, 0.0, 0.69]),            
            # position=np.array([0.0, 0.0, 0.60]),                        
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            kinematic=True,
        )

        self._products: List = self.create_products()

    @property
    def products(self) -> List[XFormPrim]:
        """
        product list

        Returns:
            List[XFormPrim]: product list
        """
        return self._products

    @property
    def table(self) -> XFormPrim:
        """
        棚

        Returns:
            XFormPrim: 棚
        """
        return self._table

    def create_products(self):
        def to_quat(r: R):
            q = r.as_quat()
            q[0], q[3] = q[3], q[0]
            return q

        products = []
        object_info = ObjectInfo("ycb_conveni_v1")
        for object_id, (name, usd_file, mass) in enumerate(object_info):
            xy = np.array([0.15, 0.15]) * (np.random.random(2) - 0.5)
            # z = 0.75 + 0.25 * np.random.random()
            z = 0.1
            # theta = 180 * np.random.random()
            # phi = 360 * np.random.random()
            # axis = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
            # angle = 360 * np.random.random()
            # orientation = to_quat(R.from_rotvec(axis * angle, degrees=True))
            products.append(
                create_convex_rigid_body(
                    name=name,
                    prim_path=f"/World/object{object_id}",
                    usd_path=usd_file,
                    mass=mass,
                    static_friction=0.5,
                    dynamic_friction=0.3,
                    position=np.array(
                        object=[xy[0], xy[1], z],
                    ),
                    kinematic=False,
                )
            )

        return products
