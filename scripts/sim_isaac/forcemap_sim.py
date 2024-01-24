from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import re
from abc import ABCMeta, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omni
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import pandas as pd
import scipy
from core.object_loader import ObjectInfo
from core.utils import message
from force_estimation import forcemap
from omni.isaac.core import World
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.sensor import Camera, ContactSensor, _sensor
from omni.physx.scripts import utils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


def set_translate(prim, new_loc):
    properties = prim.GetPropertyNames()
    if "xformOp:translate" in properties:
        translate_attr = prim.GetAttribute("xformOp:translate")
        translate_attr.Set(new_loc)
    elif "xformOp:translation" in properties:
        translation_attr = prim.GetAttribute("xformOp:translate")
        translation_attr.Set(new_loc)
    elif "xformOp:transform" in properties:
        transform_attr = prim.GetAttribute("xformOp:transform")
        matrix = prim.GetAttribute("xformOp:transform").Get()
        matrix.SetTranslateOnly(new_loc)
        transform_attr.Set(matrix)
    else:
        xform = UsdGeom.Xformable(prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(Gf.Matrix4d().SetTranslate(new_loc))


def read_contact_sensor(contact_sensor):
    csif = _sensor.acquire_contact_sensor_interface()
    cs_raw_data = csif.get_contact_sensor_raw_data(contact_sensor.prim_path)
    backend_utils = contact_sensor._backend_utils
    device = contact_sensor._device

    contacts = []
    for i in range(len(cs_raw_data)):
        contact_point = dict()
        contact_point["body0"] = csif.decode_body_name(int(cs_raw_data["body0"][i]))
        contact_point["body1"] = csif.decode_body_name(int(cs_raw_data["body1"][i]))

        contact_point["position"] = backend_utils.create_tensor_from_list(
            [
                cs_raw_data["position"][i][0],
                cs_raw_data["position"][i][1],
                cs_raw_data["position"][i][2],
            ],
            dtype="float32",
            device=device,
        )
        contact_point["normal"] = backend_utils.create_tensor_from_list(
            [cs_raw_data["normal"][i][0], cs_raw_data["normal"][i][1], cs_raw_data["normal"][i][2]],
            dtype="float32",
            device=device,
        )
        contact_point["impulse"] = backend_utils.create_tensor_from_list(
            [
                cs_raw_data["impulse"][i][0],
                cs_raw_data["impulse"][i][1],
                cs_raw_data["impulse"][i][2],
            ],
            dtype="float32",
            device=device,
        )
        contacts.append(contact_point)

    current_frame = {}
    current_frame["contacts"] = contacts
    if len(contacts) > 0:
        current_frame["in_contact"] = True
    else:
        current_frame["in_contact"] = False
    return current_frame
    # return contact_sensor.get_current_frame()


class Object:
    def __init__(self, usd_file, object_id, mass, world, geom_prim):
        self._id = object_id
        self._geom_prim = geom_prim
        self._usd_file = usd_file
        massAPI = UsdPhysics.MassAPI.Apply(geom_prim.prim)
        print("SET_MASS: ", self.get_name(), mass)
        massAPI.CreateMassAttr().Set(mass)

        self.attach_contact_sensor(world)
        self.add_material()

    def get_geom_primitive(self):
        return self._geom_prim

    def get_primitive(self):
        return self.get_geom_primitive().prim

    def get_primitive_name(self):
        return self.get_geom_primitive().prim.GetPath()

    def get_name(self):
        return self.get_geom_primitive().name

    def get_ID(self):
        return self._id

    def get_contact_sensor(self):
        return self._contact_sensor

    def attach_contact_sensor(self, world):
        self._contact_sensor = world.scene.add(
            ContactSensor(
                prim_path="/World/object{}/contact_sensor".format(self._id),
                name=f"{self._id}_cs",
                min_threshold=0,
                max_threshold=10000000,
                radius=1,
                translation=np.array([0, 0, 0]),
            )
        )
        self._contact_sensor.add_raw_contact_data_to_frame()

    def add_material(self):
        self.get_geom_primitive().apply_physics_material(
            PhysicsMaterial(
                prim_path="/World/object{}/PhysicsMaterial".format(self._id),
                name=f"{self._id}_pm",
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            )
        )


class ForcemapSim:
    def __init__(self) -> None:
        self._loaded_objects = []
        self._used_objects = []
        self._object_inv_table = {}
        self._number_of_lights = 3

    @abstractmethod
    def update_scene(self):
        """
        Create a new scene
        """
        pass

    def get_active_objects(self):
        return self._used_objects

    def get_object_name_by_primitive_name(self, prim_name):
        return self._object_inv_table[prim_name]

    def sample_object_pose(self):
        pos_xy = np.array([0.10, 0.0]) + np.array([0.2, 0.3]) * (np.random.random(2) - 0.5)
        pos_z = 0.76 + 0.3 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [pos_xy[0], pos_xy[1], pos_z], (axis, angle)

    def place_objects(self, n):
        self._used_objects = []
        for i, o in enumerate(np.random.choice(self._loaded_objects, n, replace=False)):
            print(f"placing {o.get_name()}")
            geom_prim = o.get_geom_primitive()
            contact_sensor = o.get_contact_sensor()

            while True:
                pose = self.sample_object_pose()
                position, (axis, angle) = pose
                orientation = rot_utils.rotvecs_to_quats(np.array(axis) * angle, degrees=True)
                geom_prim.set_world_pose(position, orientation)

                no_collision = True
                for j in range(3):
                    self.wait_for_stability(n_steps=1)
                    cs_current_frame = read_contact_sensor(contact_sensor)
                    if cs_current_frame["in_contact"]:
                        no_collision = False
                if no_collision:
                    break

            print(f"{o.get_name()} placed")
            self._used_objects.append(o)

    def wait_for_stability(self, n_steps=100, render=True):
        stable = True
        for n in range(n_steps):
            self.get_world().step(render=render)
        return stable

    def add_camera_D415(self):
        camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.4, 0.0, 1.15]),
            frequency=20,
            resolution=(1280, 720),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 45, 180]), degrees=True),
        )
        camera.initialize()  # This is needed after Isaac 2023
        camera.set_focal_length(1.88)
        camera.set_focus_distance(40)
        camera.set_horizontal_aperture(2.7288)
        camera.set_vertical_aperture(1.5498)
        camera.set_clipping_range(0.1, 5.0)

        camera.initialize()
        self._camera = camera

    def get_camera(self):
        return self._camera

    def add_lights(self):
        for i in range(self._number_of_lights):
            # This doesn't work after Isaac 2023
            # prim_utils.create_prim(
            #     f"/World/Light/SphereLight{i}",
            #     "SphereLight",
            #     translation=(0.0, 0.0, 3.0),
            #     attributes={"radius": 1.0, "intensity": 5000.0, "color": (0.8, 0.8, 0.8)},
            # )
            light = UsdLux.SphereLight.Define(self._world.stage, Sdf.Path(f"/World/Light/SphereLight{i}"))
            light.CreateRadiusAttr(1.0)
            light.CreateIntensityAttr(5000.0)
            light.CreateColorAttr((0.8, 0.8, 0.8))
            XFormPrim(str(light.GetPath().pathString)).set_world_pose([0.0, 0.0, 3.0])

    def randomize_lights(self):
        for i in range(self._number_of_lights):
            prim = self._world.stage.GetPrimAtPath(f"/World/Light/SphereLight{i}")
            pos_xy = np.array([-2.5, -2.5]) + 5.0 * np.random.random(2)
            pos_z = 2.0 + 1.5 * np.random.random()
            set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))
            prim.GetAttribute("intensity").Set(2000.0 + 10000.0 * np.random.random())
            prim.GetAttribute("color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))
            # prim.GetAttribute("scale").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))

    def randomize_camera_parameters(self):
        d = 0.7 + 0.2 * (np.random.random() - 0.5)
        theta = 20 + 50 * np.random.random()
        phi = -50 + 100 * np.random.random()
        th = np.deg2rad(theta)
        ph = np.deg2rad(phi)
        position = np.array([0, 0, 0.75]) + d * np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
        theta2 = theta + 6 * (np.random.random() - 0.5)
        phi2 = phi + 6 * (np.random.random() - 0.5)
        orientation = rot_utils.euler_angles_to_quats(np.array([0, theta2, 180 + phi2]), degrees=True)
        print(f"position={position}, orientation={orientation}")
        self._camera.set_world_pose(position, orientation)

        self._camera.set_focal_length(1.88 + 0.1128 * (np.random.random() - 0.5))  # 3% of the spec
        self._camera.set_focus_distance(50)
        self._camera.set_horizontal_aperture(2.6034 + 0.1562 * (np.random.random() - 0.5))
        self._camera.set_vertical_aperture(1.4621 + 0.0877 * (np.random.random() - 0.5))

    def randomize_object_colors(self):
        for o in self._used_objects:
            object_id = o.get_ID()
            mtl_prim = self._world.stage.GetPrimAtPath(f"/World/object{object_id}/Looks/material_0")
            # mtl_prim.GetAttribute("color tint").Set()
            omni.usd.create_material_input(
                mtl_prim, "diffuse_tint", Gf.Vec3f(*(0.5 + 0.5 * np.random.random(3))), Sdf.ValueTypeNames.Color3f
            )
            mtl_shade = UsdShade.Material(mtl_prim)
            obj_prim = self._world.stage.GetPrimAtPath(f"/World/object{object_id}")
            UsdShade.MaterialBindingAPI(obj_prim).Bind(mtl_shade, UsdShade.Tokens.strongerThanDescendants)

    def create_primitive(self, usd_path, prim_path, name):
        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        geom_prim = GeometryPrim(prim_path=prim_path, name=name, collision=True)
        self._world.scene.add(geom_prim)
        return geom_prim

    def setup_scene(self, scene_usd_file, obj_config):
        self._obj_config = obj_config
        create_new_stage()
        # self._world = World(**self._world_settings)
        self._world = World()
        # self._physics_context = PhysicsContext()
        # self._physics_context.enable_gpu_dynamics(flag=True)
        # self._world.scene.add_default_ground_plane()

        # world.scene.add(XFormPrim(prim_path="/World/collisionGroups", name="collision_groups_xform"))

        # add environment
        geom_prim = self.create_primitive(scene_usd_file, "/World/env", "env_geom")
        geom_prim.set_world_pose(position=[0, 0, 0])

        # add objects
        for object_id, (name, usd_file, mass) in enumerate(self._obj_config):
            prim_path = f"/World/object{object_id}"
            geom_prim = self.create_primitive(usd_file, prim_path, name)

            # geom_prim.set_collision_approximation("convexDecomposition")
            utils.setRigidBody(geom_prim.prim.GetPrim(), "convexDecomposition", False)
            # utils.setRigidBody(geom_prim.prim.GetPrim(), "sdfMesh", False)

            # prim = geom_prim.prim.GetPrim()
            # meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
            # meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            # meshCollision.CreateSdfResolutionAttr().Set(256)
            # meshcollisionAPI.CreateApproximationAttr().Set("sdf")
            self._loaded_objects.append(Object(usd_file, object_id, mass, self._world, geom_prim))

        for o in self._loaded_objects:
            self._object_inv_table[str(o.get_primitive_name())] = o.get_name()

        self.reset_object_positions()

        self.add_walls()

        self.add_lights()
        self.add_camera_D415()

        self._setup_simulation()

    def reset_object_positions(self):
        for i, o in enumerate(self._loaded_objects):
            geom_prim = o.get_geom_primitive()
            geom_prim.set_world_pose(position=[0.1 * i, 0, 0])
        self._used_objects = []

    def change_observation_condition(self):
        # self.randomize_lights()
        self.randomize_camera_parameters()
        self.randomize_object_colors()

    def stop_simulation(self):
        self._world.stop()

    def add_walls(self):
        self._walls = []
        scale = [0.05, 0.6, 0.4]
        color = [0.5, 0.5, 0.5]
        positions = [[0.25, 0.0, 0.8], [-0.25, 0.0, 0.8], [0.0, 0.3, 0.8], [0.0, -0.3, 0.8]]
        orientations = [[0, 0, 0], [0, 0, 0], [0, 0, np.pi / 2], [0, 0, np.pi / 2]]
        for i in range(4):
            w = self.get_world().scene.add(
                FixedCuboid(
                    name=f"wall{i}",
                    position=np.array(positions[i]),
                    orientation=rot_utils.euler_angles_to_quats(np.array(orientations[i])),
                    prim_path=f"/World/wall{i}",
                    size=1.0,
                    scale=np.array(scale),
                    color=np.array(color),
                )
            )
            w.set_visibility(True)
            self._walls.append(w)

    # async def _add_table(self):
    #     ##shop_table
    #     self._table_ref_geom = self._world.scene.get_object(f"table_ref_geom")
    #     self._table_ref_geom.set_local_scale(np.array([self._table_scale]))
    #     self._table_ref_geom.set_world_pose(position=self._table_position)
    #     self._table_ref_geom.set_default_state(position=self._table_position)
    #     lb = self._world.scene.compute_object_AABB(name=f"table_ref_geom")
    #     zmin = lb[0][2]
    #     zmax = lb[1][2]
    #     self._table_position[2] = -zmin
    #     self._table_height = zmax
    #     self._table_ref_geom.set_collision_approximation("none")
    #     self._convexIncludeRel.AddTarget(self._table_ref_geom.prim_path)

    #     ##tooling_plate
    #     self._tooling_plate_geom = self._world.scene.get_object(f"tooling_plate_geom")
    #     self._tooling_plate_geom.set_local_scale(np.array([self._table_scale]))
    #     lb = self._world.scene.compute_object_AABB(name=f"tooling_plate_geom")
    #     zmin = lb[0][2]
    #     zmax = lb[1][2]
    #     tooling_transform = self._tooling_plate_offset
    #     tooling_transform[2] = -zmin + self._table_height
    #     tooling_transform = tooling_transform + self._table_position
    #     self._tooling_plate_geom.set_world_pose(position=tooling_transform)
    #     self._tooling_plate_geom.set_default_state(position=tooling_transform)
    #     self._tooling_plate_geom.set_collision_approximation("boundingCube")
    #     self._table_height += zmax - zmin
    #     self._convexIncludeRel.AddTarget(self._tooling_plate_geom.prim_path)

    #     ##pipe
    #     self._pipe_geom = self._world.scene.get_object(f"pipe_geom")
    #     self._pipe_geom.set_local_scale(np.array([self._table_scale]))
    #     lb = self._world.scene.compute_object_AABB(name=f"pipe_geom")
    #     zmin = lb[0][2]
    #     zmax = lb[1][2]
    #     self._pipe_height = zmax - zmin
    #     pipe_transform = self._pipe_pos_on_table
    #     pipe_transform[2] = -zmin + self._table_height
    #     pipe_transform = pipe_transform + self._table_position
    #     self._pipe_geom.set_world_pose(position=pipe_transform, orientation=np.array([0, 0, 0, 1]))
    #     self._pipe_geom.set_default_state(position=pipe_transform, orientation=np.array([0, 0, 0, 1]))
    #     self._pipe_geom.set_collision_approximation("none")
    #     self._convexIncludeRel.AddTarget(self._pipe_geom.prim_path)
    #     await self._world.reset_async()

    # async def _add_vibra_table(self):
    #     self._vibra_table_bottom_geom = self._world.scene.get_object(f"vibra_table_bottom_geom")
    #     self._vibra_table_bottom_geom.set_local_scale(np.array([self._table_scale]))
    #     lb = self._world.scene.compute_object_AABB(name=f"vibra_table_bottom_geom")
    #     zmin = lb[0][2]
    #     bot_part_pos = self._vibra_table_position_offset
    #     bot_part_pos[2] = -zmin + self._table_height
    #     bot_part_pos = bot_part_pos + self._table_position
    #     self._vibra_table_bottom_geom.set_world_pose(position=bot_part_pos)
    #     self._vibra_table_bottom_geom.set_default_state(position=bot_part_pos)
    #     self._vibra_table_bottom_geom.set_collision_approximation("none")
    #     self._convexIncludeRel.AddTarget(self._vibra_table_bottom_geom.prim_path)

    #     # clamps
    #     self._vibra_table_clamps_geom = self._world.scene.get_object(f"vibra_table_clamps_geom")
    #     self._vibra_table_clamps_geom.set_collision_approximation("none")
    #     self._convexIncludeRel.AddTarget(self._vibra_table_clamps_geom.prim_path)

    #     # vibra_table
    #     self._vibra_table_xform = self._world.scene.get_object(f"vibra_table_xform")
    #     self._vibra_table_position = bot_part_pos
    #     vibra_kinematic_prim = self._vibra_table_xform.prim
    #     rbApi = UsdPhysics.RigidBodyAPI.Apply(vibra_kinematic_prim.GetPrim())
    #     rbApi.CreateRigidBodyEnabledAttr(True)
    #     rbApi.CreateKinematicEnabledAttr(True)

    #     # visual
    #     self._vibra_table_visual_xform = self._world.scene.get_object(f"vibra_table_visual_xform")
    #     self._vibra_table_visual_xform.set_world_pose(position=self._vibra_top_offset)
    #     self._vibra_table_visual_xform.set_default_state(position=self._vibra_top_offset)
    #     self._vibra_table_visual_xform.set_local_scale(np.array([self._table_scale]))

    #     # collision
    #     self._vibra_table_collision_ref_geom = self._world.scene.get_object(f"vibra_table_collision_ref_geom")
    #     offset = self._vibra_top_offset + self._vibra_table_top_to_collider_offset
    #     self._vibra_table_collision_ref_geom.set_local_scale(np.array([1.0]))
    #     self._vibra_table_collision_ref_geom.set_world_pose(position=offset)
    #     self._vibra_table_collision_ref_geom.set_default_state(position=offset)
    #     self._vibra_table_collision_ref_geom.apply_physics_material(self._vibra_table_physics_material)
    #     self._convexIncludeRel.AddTarget(self._vibra_table_collision_ref_geom.prim_path)
    #     self._vibra_table_collision_ref_geom.set_collision_approximation("convexHull")
    #     vibra_kinematic_prim.SetInstanceable(True)
    #     self._vibra_table_xform.set_world_pose(position=self._vibra_table_position, orientation=np.array([0, 0, 0, 1]))
    #     self._vibra_table_xform.set_default_state(
    #         position=self._vibra_table_position, orientation=np.array([0, 0, 0, 1])
    #     )
    #     self._vibra_table_visual_xform.set_default_state(
    #         position=self._vibra_table_visual_xform.get_world_pose()[0],
    #         orientation=self._vibra_table_visual_xform.get_world_pose()[1],
    #     )
    #     self._vibra_table_collision_ref_geom.set_default_state(
    #         position=self._vibra_table_collision_ref_geom.get_world_pose()[0],
    #         orientation=self._vibra_table_collision_ref_geom.get_world_pose()[1],
    #     )
    #     await self._world.reset_async()

    # async def _add_nuts_and_bolt(self, add_debug_nut=False):
    #     angle_delta = np.pi * 2.0 / self._num_bolts
    #     for j in range(self._num_bolts):
    #         self._bolt_geom = self._world.scene.get_object(f"bolt{j}_geom")
    #         self._bolt_geom.prim.SetInstanceable(True)
    #         bolt_pos = np.array(self._pipe_pos_on_table) + self._table_position
    #         bolt_pos[0] += np.cos(j * angle_delta) * self._bolt_radius
    #         bolt_pos[1] += np.sin(j * angle_delta) * self._bolt_radius
    #         bolt_pos[2] = self._bolt_z_offset_to_pipe + self._table_height
    #         self._bolt_geom.set_world_pose(position=bolt_pos)
    #         self._bolt_geom.set_default_state(position=bolt_pos)
    #         self._boltMeshIncludeRel.AddTarget(self._bolt_geom.prim_path)
    #         self._bolt_geom.apply_physics_material(self._bolt_physics_material)

    #     await self._generate_nut_initial_poses()
    #     for nut_idx in range(self._num_nuts):
    #         nut_pos = self._nut_init_poses[nut_idx, :3].copy()
    #         if add_debug_nut and nut_idx == 0:
    #             nut_pos[0] = 0.78
    #             nut_pos[1] = self._vibra_table_nut_pickup_pos_offset[1] + self._vibra_table_position[1]  # 0.0264
    #         if add_debug_nut and nut_idx == 1:
    #             nut_pos[0] = 0.78
    #             nut_pos[1] = 0.0264 - 0.04

    #         self._nut_geom = self._world.scene.get_object(f"nut{nut_idx}_geom")
    #         self._nut_geom.prim.SetInstanceable(True)
    #         self._nut_geom.set_world_pose(position=np.array(nut_pos.tolist()))
    #         self._nut_geom.set_default_state(position=np.array(nut_pos.tolist()))
    #         physxRBAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim)
    #         physxRBAPI.CreateSolverPositionIterationCountAttr().Set(self._solverPositionIterations)
    #         physxRBAPI.CreateSolverVelocityIterationCountAttr().Set(self._solverVelocityIterations)
    #         self._nut_geom.apply_physics_material(self._nut_physics_material)
    #         self._convexIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_Convex")
    #         self._nutMeshIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_SDF")
    #         rbApi3 = UsdPhysics.RigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
    #         rbApi3.CreateRigidBodyEnabledAttr(True)
    #         physxAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
    #         physxAPI.CreateSleepThresholdAttr().Set(0.0)
    #         massAPI = UsdPhysics.MassAPI.Apply(self._nut_geom.prim.GetPrim())
    #         massAPI.CreateMassAttr().Set(self._mass_nut)
    #     await self._world.reset_async()

    # async def _generate_nut_initial_poses(self):
    #     self._nut_init_poses = np.zeros((self._num_nuts, 7), dtype=np.float32)
    #     self._nut_init_poses[:, -1] = 1  # quat to identity
    #     nut_spiral_center = self._vibra_table_position + self._nut_spiral_center_vibra_offset
    #     nut_spiral_center += self._vibra_top_offset
    #     for nut_idx in range(self._num_nuts):
    #         self._nut_init_poses[nut_idx, :3] = np.array(nut_spiral_center)
    #         self._nut_init_poses[nut_idx, 0] += self._nut_radius * np.sin(
    #             np.pi / 3.0 * nut_idx
    #         ) + self._nut_dist_delta * (nut_idx // 6)
    #         self._nut_init_poses[nut_idx, 1] += self._nut_radius * np.cos(
    #             np.pi / 3.0 * nut_idx
    #         ) + self._nut_dist_delta * (nut_idx // 6)
    #         self._nut_init_poses[nut_idx, 2] += self._nut_height_delta * (nut_idx // 6)
    #         if self._randomize_nut_positions:
    #             self._nut_init_poses[nut_idx, 0] += self._rng.uniform(
    #                 -self._nut_position_noise_minmax, self._nut_position_noise_minmax
    #             )
    #             self._nut_init_poses[nut_idx, 1] += self._rng.uniform(
    #                 -self._nut_position_noise_minmax, self._nut_position_noise_minmax
    #             )
    #     await self._world.reset_async()

    def _setup_simulation(self):
        self._world.initialize_physics()
        self._world.reset()
        self._world.play()

    #     self._scene = PhysicsContext()
    #     self._scene.set_solver_type(self._solver_type)
    #     self._scene.set_broadphase_type("GPU")
    #     self._scene.enable_gpu_dynamics(flag=True)
    #     self._scene.set_friction_offset_threshold(0.01)
    #     self._scene.set_friction_correlation_distance(0.0005)
    #     self._scene.set_gpu_total_aggregate_pairs_capacity(10 * 1024)
    #     self._scene.set_gpu_found_lost_pairs_capacity(10 * 1024)
    #     self._scene.set_gpu_heap_capacity(64 * 1024 * 1024)
    #     self._scene.set_gpu_found_lost_aggregate_pairs_capacity(10 * 1024)

    #     self._meshCollisionGroup = UsdPhysics.CollisionGroup.Define(
    #         self._world.scene.stage, "/World/collisionGroups/meshColliders"
    #     )
    #     collectionAPI = Usd.CollectionAPI.ApplyCollection(self._meshCollisionGroup.GetPrim(), "colliders")
    #     self._nutMeshIncludeRel = collectionAPI.CreateIncludesRel()
    #     self._convexCollisionGroup = UsdPhysics.CollisionGroup.Define(
    #         self._world.scene.stage, "/World/collisionGroups/convexColliders"
    #     )
    #     collectionAPI = Usd.CollectionAPI.ApplyCollection(self._convexCollisionGroup.GetPrim(), "colliders")
    #     self._convexIncludeRel = collectionAPI.CreateIncludesRel()
    #     self._boltCollisionGroup = UsdPhysics.CollisionGroup.Define(
    #         self._world.scene.stage, "/World/collisionGroups/boltColliders"
    #     )
    #     collectionAPI = Usd.CollectionAPI.ApplyCollection(self._boltCollisionGroup.GetPrim(), "colliders")
    #     self._boltMeshIncludeRel = collectionAPI.CreateIncludesRel()

    #     # invert group logic so only groups that filter each-other will collide:
    #     self._scene.set_invert_collision_group_filter(True)

    #     # # the SDF mesh collider nuts should only collide with the bolts
    #     filteredRel = self._meshCollisionGroup.CreateFilteredGroupsRel()
    #     filteredRel.AddTarget("/World/collisionGroups/boltColliders")

    #     # # the convex hull nuts should collide with each other, the vibra table, and the grippers
    #     filteredRel = self._convexCollisionGroup.CreateFilteredGroupsRel()
    #     filteredRel.AddTarget("/World/collisionGroups/convexColliders")

    #     # # the SDF mesh bolt only collides with the SDF mesh nut colliders
    #     filteredRel = self._boltCollisionGroup.CreateFilteredGroupsRel()
    #     filteredRel.AddTarget("/World/collisionGroups/meshColliders")

    def world_cleanup(self):
        self._controller = None
        return

    def get_world(self):
        return self._world


class RandomSeriaBasketSim(ForcemapSim):
    def __init__(self):
        # ycb_piled_scene.usd
        super().__init__()

    def update_scene(self):
        self._world.reset()
        number_of_objects = np.clip(np.random.poisson(5), 1, 8)
        self.place_objects(number_of_objects)

    def sample_object_pose(self):
        xy = np.array([0.15, 0.08]) * (np.random.random(2) - 0.5)
        z = 0.75 + 0.25 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [xy[0], xy[1], z], (axis, angle)

    def randomize_camera_parameters(self):
        position = np.array([0, 0, 1.358]) + 0.02 * (np.random.random(3) - 0.5)
        orientation = rot_utils.euler_angles_to_quats(
            np.array([0, 90, 180] + 6 * (np.random.random(3) - 0.5)), degrees=True
        )
        self._camera.set_world_pose(position, orientation)


class RandomTabletopSim(ForcemapSim):
    def __init__(self, multiviews=False):
        self._multiviews = multiviews
        super().__init__()

    def update_scene(self):
        self._world.reset()
        number_of_objects = np.clip(np.random.poisson(9), 4, 10)
        self.place_objects(number_of_objects)

    def sample_object_pose(self):
        xy = np.array([0.15, 0.15]) * (np.random.random(2) - 0.5)
        z = 0.75 + 0.25 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [xy[0], xy[1], z], (axis, angle)

    def randomize_camera_parameters(self):
        if self._multiviews:
            d = 0.7 + 0.2 * (np.random.random() - 0.5)
            theta0 = np.radians(20)
            theta1 = np.radians(60)
            z = np.random.random()
            th = np.arcsin(z * (np.sin(theta1) - np.sin(theta0)) + np.sin(theta0))
            phi = 360 * np.random.random()
            ph = np.deg2rad(phi)
            position = np.array([0, 0, 0.75]) + d * np.array(
                [np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)]
            )
            theta2 = np.degrees(th) + 6 * (np.random.random() - 0.5)
            phi2 = phi + 6 * (np.random.random() - 0.5)
            orientation = rot_utils.euler_angles_to_quats(np.array([0, theta2, 180 + phi2]), degrees=True)
            print(f"position={position}, orientation={orientation}")
            self._camera.set_world_pose(position, orientation)
            self._camera.set_focal_length(1.88 + 0.1128 * (np.random.random() - 0.5))  # 3% of the spec
            self._camera.set_focus_distance(50)
            self._camera.set_horizontal_aperture(2.6034 + 0.1562 * (np.random.random() - 0.5))
            self._camera.set_vertical_aperture(1.4621 + 0.0877 * (np.random.random() - 0.5))
        else:  # perturb viewpoint
            # position = np.array([0, 0, 1.358]) + 0.02 * (np.random.random(3) - 0.5)
            position = np.array([0, 0, 1.6]) + 0.02 * (np.random.random(3) - 0.5)
            orientation = rot_utils.euler_angles_to_quats(
                np.array([0, 90, 180] + 3 * (np.random.random(3) - 0.5)), degrees=True
            )
            self._camera.set_world_pose(position, orientation)


class AllObjectTabletopSim(ForcemapSim):
    def __init__(self, world, conf):
        # self._asset_path = os.environ["HOME"] + "/Dataset/scenes/green_table_scene.usd"
        super().__init__()

    def update_scene(self):
        self._world.reset()
        self.place_objects()

    def sample_object_pose(self):
        xy = np.array([0.15, 0.15]) * (np.random.random(2) - 0.5)
        z = 0.75 + 0.25 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [xy[0], xy[1], z], (axis, angle)

    def place_objects(self):
        self._used_objects = []
        for i, o in enumerate(self._loaded_objects):
            print(f"placing {o.get_name()}")
            prim = o.get_primitive()
            contact_sensor = o.get_contact_sensor()

            while True:
                pose = self.sample_object_pose()
                set_pose(prim, pose)

                no_collision = True
                for j in range(3):
                    self.wait_for_stability(n_steps=1)
                    cs_current_frame = read_contact_sensor(contact_sensor)
                    if cs_current_frame["in_contact"]:
                        no_collision = False
                if no_collision:
                    break

            print(f"{o.get_name()} placed")
            self._used_objects.append(o)

    def randomize_camera_parameters(self):
        d = 1.0 + 0.2 * (np.random.random() - 0.5)
        theta0 = np.radians(20)
        theta1 = np.radians(60)
        z = np.random.random()
        th = np.arcsin(z * (np.sin(theta1) - np.sin(theta0)) + np.sin(theta0))
        phi = 360 * np.random.random()
        ph = np.deg2rad(phi)
        position = np.array([0, 0, 0.75]) + d * np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
        theta2 = np.degrees(th) + 6 * (np.random.random() - 0.5)
        phi2 = phi + 6 * (np.random.random() - 0.5)
        orientation = rot_utils.euler_angles_to_quats(np.array([0, theta2, 180 + phi2]), degrees=True)
        print(f"position={position}, orientation={orientation}")
        self._camera.set_world_pose(position, orientation)

        self._camera.set_focal_length(1.88 + 0.1128 * (np.random.random() - 0.5))  # 3% of the spec
        self._camera.set_focus_distance(50)
        self._camera.set_horizontal_aperture(2.6034 + 0.1562 * (np.random.random() - 0.5))
        self._camera.set_vertical_aperture(1.4621 + 0.0877 * (np.random.random() - 0.5))

    def randomize_object_colors(self):
        pass


# class AlignedToppoSim(ForcemapSim):
#     def __init__(self):
#         self.create_toppos(20)

#     def create_toppos(self, n):
#         self._loaded_objects = []
#         usd_file = os.path.join(
#             os.environ["HOME"],
#             "Dataset/Konbini/VER002/Seamless/vt2048/15_TOPPO-ART-vt2048SA/15_TOPPO-ART-pl2048SA/15_TOPPO-ART-pl2048SA.usd",
#         )
#         for i in range(n):
#             self._loaded_objects.append(Object(usd_file, i, self._world))

#     def create_toppo_scene(self, n):
#         global used_objects
#         used_objects = []

#         world.reset()
#         for i, o in enumerate(loaded_objects):
#             prim = o.get_primitive()
#             pos_xy = np.array([0.0, -0.1]) + np.array([0.0, 0.025]) * i
#             pos_z = 0.73

#             axis = [1, 0, 0]
#             angle = 90
#             set_pose(prim, ([pos_xy[0], pos_xy[1], pos_z], (axis, angle)))
#             used_objects.append(o)

#         # randomize_lights()
#         # randomize_camera_parameters()
#         # randomize_object_colors()
#         # wait_for_stability(n_steps=1200)


class Recorder:
    def __init__(self, sim, output_force_distribution=False):
        self._output_force_distribution = output_force_distribution
        self._frameNo = 0
        self._sim = sim
        self._data_dir = "data"

    def get_object_name(self, name):
        if name == "/World/env" or re.match("/World/wall.*", name) != None:
            return None
        if name == "/World/env/table":
            return "table"
        return self._sim.get_object_name_by_primitive_name(name)

    def save_state(self):
        """
        Record bin-state and force
        """
        contact_positions = []
        impulse_values = []
        contacting_objects = []
        bin_state = []

        for o in self._sim.get_active_objects():
            pose = omni.usd.get_world_transform_matrix(o.get_primitive())
            trans = pose.ExtractTranslation()
            trans = np.array(trans)
            quat = pose.ExtractRotation().GetQuaternion()
            quat = np.array(list(quat.imaginary) + [quat.real])

            contact_sensor = o.get_contact_sensor()
            cs_current_frame = read_contact_sensor(contact_sensor)

            print(
                f"SCENE[{self._frameNo}], ",
                o.get_name(),
                f"# of contacts={len(cs_current_frame['contacts'])}",
                trans,
                quat,
            )

            for contact in cs_current_frame["contacts"]:
                print("CC: ", contact)
                objectA = self.get_object_name(contact["body0"])
                objectB = self.get_object_name(contact["body1"])
                # print(f"C: {contact['body0']}, {contact['body1']}", end="  ")
                # print(f"C: {objectA}, {objectB}", end="  ")

                # if objectA == None or objectB == None:
                #     continue

                # if scipy.linalg.norm(contact["impulse"]) < 1e-12:
                #     print(f"skip, too small impulse: {contact['impulse']}")
                #     continue

                # contact_positions.append(contact["position"])
                # impulse_values.append(scipy.linalg.norm(contact["impulse"]))
                # contacting_objects.append((objectA, objectB))

            bin_state.append((o.get_name(), (trans, quat)))

        pd.to_pickle(bin_state, os.path.join(self._data_dir, "bin_state{:05d}.pkl".format(self._frameNo)))
        pd.to_pickle(
            (contact_positions, impulse_values, contacting_objects),
            os.path.join(self._data_dir, "contact_raw_data{:05d}.pkl".format(self._frameNo)),
        )

        if self._output_force_distribution:
            force_dist = self.convert_to_force_distribution(contact_positions, impulse_values, contacting_objects)
            pd.to_pickle(force_dist, os.path.join(self._data_dir, "force_zip{:05d}.pkl".format(self._frameNo)))

    def save_image(self, viewNum=None, crop_center=True):
        rgb = self._sim.get_camera().get_rgba()[:, :, :3]
        rgb_path = os.path.join(self._data_dir, f"rgb{self._frameNo:05}_{viewNum:05}.jpg")
        if crop_center:
            cam_height, cam_width, _ = rgb.shape
            height = 360
            width = 512
            rgb_cropped = rgb[
                int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
                int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
            ]
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pd.to_pickle(
            self._sim.get_camera().get_world_pose(),
            os.path.join(self._data_dir, f"camera_info{self._frameNo:05}_{viewNum:05}.pkl"),
        )

    def incrementFrameNumber(self):
        self._frameNo += 1

    # def get_bin_state(self):
    #     prim_path = "/World/_09_gelatin_box/contact_sensor"
    #     obj = world.scene.get_object(prim_path)
    #     return obj.get_world_pose()

    def convert_to_force_distribution(self, contact_positions, impulse_values, log_scale=False):
        fmap = forcemap.GridForceMap("small_table")
        d = fmap.getDensity(contact_positions, impulse_values)
        if log_scale:
            d = np.log(1 + d)
        return d


class DatasetGenerator:
    def __init__(self, sim, output_force_distribution=True):
        self._sim = sim
        self._recorder = Recorder(sim, output_force_distribution=output_force_distribution)

    def generate(self, number_of_scenes, number_of_views=5):
        for frameNum in range(number_of_scenes):
            self._sim.update_scene()
            self._sim.wait_for_stability()

            for viewNum in range(number_of_views):
                self._sim.change_observation_condition()
                self._sim.wait_for_stability(n_steps=20)
                self._recorder.save_image(viewNum)
            self._recorder.save_state()
            self._recorder.incrementFrameNumber()

        while simulation_app.is_running():
            self._sim.get_world().step(render=True)

        self._sim.stop_simulation()
        simulation_app.close()


# Konbini objects
# usd_files = glob.glob(os.path.join(os.environ['HOME'], '/Program/moonshot/ae_lstm/specification/meshes/objects/ycb/usd/*/*.usd'))
# names = [os.path.splitext(usd_file.split("/")[-1])[0] for usd_file in usd_files]

# Seria basket scene (IROS2023, moonshot interim demo.)
# scene = RandomSeriaBasketScene(world, conf)

# scene = RandomTableScene(world, conf)

# # scene = AllObjectTableScene(world, conf)


env_usd_file = f'{os.environ["HOME"]}/Dataset/scenes/green_table_scene.usd'
obj_config = ObjectInfo("ycb_v1")

sim = RandomTabletopSim()
sim.setup_scene(env_usd_file, obj_config)
dg = DatasetGenerator(sim, output_force_distribution=False)
# dg.generate(5, 3)


def steps(n=10, render=True):
    for i in range(n):
        sim.get_world().step(render=render)
