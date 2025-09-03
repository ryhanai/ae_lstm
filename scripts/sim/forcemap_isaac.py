# -*- coding: utf-8 -*-

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import omni
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.usd
import pandas as pd
import scipy
from force_estimation import forcemap
from dataset.object_loader import ObjectInfo
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.sensor import Camera, ContactSensor
from omni.physx.scripts import utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def set_pose(prim, pose):
    p, (axis, angle) = pose
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()  # Is this necessary?
    transform = xform.AddTransformOp()
    mat = Gf.Matrix4d()
    mat.SetTranslateOnly(Gf.Vec3d(*p))
    mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(*axis), angle))
    transform.Set(mat)


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


def set_rotate_mat(prim, rot_mat):
    xform = UsdGeom.Xformable(prim)
    xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op.Set(Gf.Matrix4d().SetRotate(rot_mat))


def set_rotate_euler(prim, rot_euler):
    """
    rot_euler : np.array
    """
    q = rot_utils.euler_angles_to_quats(rot_euler, degrees=True)
    print(q)
    set_rotate_mat(prim, Gf.Matrix3d(Gf.Quatd(q[3], q[0], q[1], q[2])))


def set_mass(prim, kg):
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(kg)
    return kg


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location=[0, 0, 0.1]):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")
    envPrim.GetReferences().AddReference(prim_usd_path)
    set_translate(envPrim, Gf.Vec3d(*location))
    return stage.GetPrimAtPath(envPrim.GetPath().pathString)


def assign_physics_material(stage, prim, prim_path):
    # geom_prim = world.scene.get_object(os.path.join(prim_path, "textured/mesh"))
    # geom_prim = stage.GetPrimAtPath(os.path.join(prim_path, "textured/mesh"))
    geom_prim = prim.geom_prim()
    geom_prim.apply_physics_material(
        PhysicsMaterial(
            prim_path=os.path.join(prim_path, "physics_material"),
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.1,
        )
    )

    # for child_prim in Usd.PrimRange(prim):
    #     if child_prim.IsA(UsdGeom.Mesh):
    #         print("SET_MATERIAL")
    #         child_prim.apply_physics_material(
    #             PhysicsMaterial(
    #                 prim_path=os.path.join(prim_env_path, "physics_material"),
    #                 static_friction=0.5,
    #                 dynamic_friction=0.4,
    #                 restitution=0.1,
    #             )
    #         )
    return prim


class Object:
    def __init__(self, name, usd_file, object_id, mass, world):
        self._usd_file = usd_file
        self._primitive_name = f"/World/object{object_id}"
        self._prim = create_prim_from_usd(world.stage, self._primitive_name, usd_file)
        # assign_physics_material(world.stage, self._prim, self._primitive_name)
        self._id = object_id
        self._name = name
        utils.setRigidBody(self._prim, "convexDecomposition", False)
        for node_prim in Usd.PrimRange(self._prim):
            for prim in Usd.PrimRange(node_prim):
                if prim.IsA(UsdGeom.Mesh):
                    print("SET_MASS: ", self._name, set_mass(prim, kg=mass))  # why is this called 3 times?

        self._contact_sensor = world.scene.add(
            ContactSensor(
                prim_path="/World/object{}/contact_sensor".format(self._id),
                name=f"{self._id}_cs",
                min_threshold=0,
                max_threshold=10000000,
                # radius=0.5,
                radius=-1,
                translation=np.array([0, 0, 0]),
            )
        )

    def get_primitive(self):
        return self._prim

    def get_primitive_name(self):
        return self._primitive_name

    def get_name(self):
        return self._name

    def get_ID(self):
        return self._id

    def get_contact_sensor(self):
        return self._contact_sensor


class Scene:
    def __init__(self, world, env_config):
        self._world = world
        self._env_config = env_config
        self._conf = ObjectInfo(env_config['object_set'])
        self._loaded_objects = []
        self._used_objects = []
        self._number_of_lights = 3

        self.create_env()
        self.create_lights()
        self._camera = self.create_camera_D415()
        self.create_objects()

    @abstractmethod
    def create_objects(self):
        """
        Load objects used for the scene
        """
        pass

    @abstractmethod
    def renew_scene(self):
        """
        Create a new scene
        """
        pass

    @abstractmethod
    def get_env_name(self):
        return "env"

    def get_active_objects(self):
        return self._used_objects

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
            prim = o.get_primitive()
            contact_sensor = o.get_contact_sensor()

            while True:
                pose = self.sample_object_pose()
                set_pose(prim, pose)

                no_collision = True
                for j in range(3):
                    self.wait_for_stability(count=1)
                    contacts = self.read_contact_sensor(contact_sensor)
                    if len(contacts) > 0:
                        no_collision = False
                if no_collision:
                    break

            print(f"{o.get_name()} placed")
            self._used_objects.append(o)

    def wait_for_stability(self, count=100):
        # stable = True
        for n in range(count):
            self._world.step(render=True)
            # for i in range(10):
            #     o = world.scene.get_object(f'object{i}')
            #     for child in o.GetChildren():
            #         child_path = child.GetPath().pathString
            #         body_handle = dc.get_rigid_body(child_path)
            #         if body_handle != 0:
            #             print(dc.get_rigid_body_linear_velocity(body_handle))
            #     # if np.linalg.norm(o.get_linear_velocity()) > 0.001:
            #     #     stable = False
        #     if stable:
        #         return True
        return False

    def create_env(self):
        create_prim_from_usd(self._world.stage, "/World/env", self._asset_path, Gf.Vec3d([0, 0, 0.0]))

    def create_camera_D415(self):
        camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.4, 0.0, 1.15]),
            frequency=20,
            resolution=(1280, 720),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 45, 180]), degrees=True),
        )
        camera.initialize()
        camera.set_focal_length(1.88)
        camera.set_focus_distance(40)
        camera.set_horizontal_aperture(2.7288)
        camera.set_vertical_aperture(1.5498)
        camera.set_clipping_range(0.1, 5.0)
        # camera.add_distance_to_image_plane_to_frame()  # Needed to measure depth        
        camera.add_distance_to_camera_to_frame()
        return camera

    def create_lights(self):
        for i in range(self._number_of_lights):
            # IsaacSim 2022.2.0
            # prim_utils.create_prim(
            #     f"/World/Light/SphereLight{i}",
            #     "SphereLight",
            #     translation=(0.0, 0.0, 3.0),
            #     attributes={"radius": 1.0, "intensity": 5000.0, "color": (0.8, 0.8, 0.8)},
            # )

            # IsaacSim > 2023
            stage = self._world.stage
            sphere_light = stage.DefinePrim(f"/World/Light/SphereLight{i}", "SphereLight")
            sphere_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(5000.0)
            if not sphere_light.HasAttribute("xformOp:translate"):
                UsdGeom.Xformable(sphere_light).AddTranslateOp()
            sphere_light.GetAttribute("xformOp:translate").Set((0, 0, 3.0))
            sphere_light.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(1.0)

    def randomize_lights(self):
        for i in range(self._number_of_lights):
            prim = self._world.stage.GetPrimAtPath(f"/World/Light/SphereLight{i}")
            pos_xy = np.array([-2.5, -2.5]) + 5.0 * np.random.random(2)
            pos_z = 2.5 + 1.5 * np.random.random()
            set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))

            # IsaacSim 2022.2.0
            # prim.GetAttribute("intensity").Set(1000.0 + 15000.0 * np.random.random())
            # prim.GetAttribute("color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))

            # IsaacSim > 2023
            prim.GetAttribute("inputs:intensity").Set(1000.0 + 15000.0 * np.random.random())
            prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))

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

    def read_contact_sensor(self, contact_sensor):
        csif = contact_sensor._contact_sensor_interface
        cs_raw_data = csif.get_contact_sensor_raw_data(contact_sensor.prim_path)
        print(f'CURRENT_FRAME: {contact_sensor.get_current_frame()}')
        print(f'{len(cs_raw_data)}: {cs_raw_data}')
        # backend_utils = simulation_context.backend_utils
        # device = simulation_context.device
        backend_utils = self._world.backend_utils
        device = self._world.device

        current_frame = []
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
            current_frame.append(contact_point)
        return current_frame


class RandomScene(Scene):
    def __init__(self, world, env_config):
        self._object_inv_table = {}
        super().__init__(world, env_config)

    def create_objects(self):
        for i, (name, usd_file, mass) in enumerate(self._conf):
            self._loaded_objects.append(Object(name, usd_file, i, mass, self._world))

        for o in self._loaded_objects:
            self._object_inv_table[o.get_primitive_name()] = o.get_name()
        self.reset_object_positions()

    def get_object_name_by_primitive_name(self, prim_name):
        try:
            return self._object_inv_table[prim_name]
        except KeyError:
            return self.get_env_name()

    def reset_object_positions(self):
        for i, o in enumerate(self._loaded_objects):
            prim = o.get_primitive()
            set_translate(prim, Gf.Vec3d(0.1 * i, 0, 0))
        self._used_objects = []

    def change_scene(self):
        self._world.reset()
        self.place_objects(10)

    def change_observation_condition(self):
        self.randomize_lights()
        self.randomize_camera_parameters()
        self.randomize_object_colors()


class RandomSeriaBasketScene(RandomScene):
    def __init__(self, world, env_config):
        self._asset_path = os.environ["HOME"] + "/Dataset/scenes/ycb_piled_scene.usd"
        super().__init__(world, env_config)

    def change_scene(self):
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

    def get_env_name(self):
        return "seria_basket"


class RandomTableScene(RandomScene):
    def __init__(self, world, env_config, multiviews=False):
        self._asset_path = f"{env_config['asset_root']}/{env_config['scene_usd']}"
        self._multiviews = multiviews
        super().__init__(world, env_config)

    def change_scene(self):
        self._world.reset()
        number_of_objects = np.clip(np.random.poisson(9), 3, 9)
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
            vp = self._env_config['viewpoint']
            rng = self._env_config['viewpoint_randomization_range']
            position = np.array(vp[0]) + rng[0] * (np.random.random(3) - 0.5)
            orientation = rot_utils.euler_angles_to_quats(
                np.array(vp[1] + rng[1] * (np.random.random(3) - 0.5)), degrees=True
            )
            self._camera.set_world_pose(position, orientation)

    def get_env_name(self):
        return "table"


class AllObjectTableScene(RandomScene):
    def __init__(self, world, conf):
        self._asset_path = os.environ["HOME"] + "/Dataset/scenes/green_table_scene.usd"
        super().__init__(world, conf)

    def change_scene(self):
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
                    self.wait_for_stability(count=1)
                    contacts = self.read_contact_sensor(contact_sensor)
                    if len(contacts) > 0:
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


class Recorder:
    def __init__(self, 
                 camera, 
                 crop_size,
                 output_image_size,
                 output_smoothed_force=False):

        self._output_smoothed_force = output_smoothed_force
        self._frameNo = 0
        self._camera = camera
        self._data_dir = "data"
        self._crop_size = crop_size
        self._output_image_size = output_image_size

    def save_state(self, scene):
        """
        Record bin-state and force
        """
        contact_positions = []
        contact_normals = []
        impulse_values = []
        contacting_objects = []
        bin_state = []

        for o in scene.get_active_objects():
            contact_sensor = o.get_contact_sensor()
            contacts = scene.read_contact_sensor(contact_sensor)

            for contact in contacts:
                # if scipy.linalg.norm(contact["impulse"]) > 1e-8:
                if True:
                    objectA = scene.get_object_name_by_primitive_name(contact["body0"])
                    objectB = scene.get_object_name_by_primitive_name(contact["body1"])
                    contact_positions.append(contact["position"])
                    contact_normals.append(contact["normal"])
                    impulse_values.append(scipy.linalg.norm(contact["impulse"]))
                    contacting_objects.append((objectA, objectB))
                    print(f' {objectA}, {objectB}, {contact["impulse"]}')

            prim = o.get_primitive()
            pose = omni.usd.get_world_transform_matrix(prim)
            trans = pose.ExtractTranslation()
            trans = np.array(trans)
            quat = pose.ExtractRotation().GetQuaternion()
            quat = np.array(list(quat.imaginary) + [quat.real])
            bin_state.append((o.get_name(), (trans, quat)))
            print(f"SCENE[{self._frameNo},{o.get_name()}]:", trans, quat)

        # remove duplicated contacts
        contact_positions, uidx = np.unique(contact_positions, axis=0, return_index=True)
        contact_normals = [contact_normals[idx] for idx in uidx]
        impulse_values = [impulse_values[idx] for idx in uidx]
        contacting_objects = [contacting_objects[idx] for idx in uidx]

        pd.to_pickle(bin_state, os.path.join(self._data_dir, "bin_state{:05d}.pkl".format(self._frameNo)))
        pd.to_pickle(
            (contact_positions, impulse_values, contacting_objects, contact_normals),
            os.path.join(self._data_dir, "contact_raw_data{:05d}.pkl".format(self._frameNo)),
        )

        if self._output_smoothed_force:
            force_dist = self.convert_to_force_distribution(contact_positions, impulse_values, contacting_objects)
            pd.to_pickle(force_dist, os.path.join(self._data_dir, "force_zip{:05d}.pkl".format(self._frameNo)))

    def save_image(self, viewNum=None):
        def crop_center_and_resize(img):
            width, height = self._crop_size
            cam_height = img.shape[0]
            cam_width = img.shape[1]
            cropped_img = img[
                int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
                int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
            ]
            return cv2.resize(cropped_img, self._output_image_size)

        rgb = self._camera.get_rgba()[:, :, :3]
        rgb_path = os.path.join(self._data_dir, f"rgb{self._frameNo:05}_{viewNum:05}.jpg")
        output_rgb = crop_center_and_resize(rgb)
        cv2.imwrite(rgb_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

        # depth = self._camera.get_depth()
        depth = self._camera.get_current_frame()["distance_to_camera"]
        depth = np.where(depth > 1., 0., depth)
        # print(f'DEPTH={depth, np.min(depth), np.max(depth), np.average(depth)}')
        depth_path = os.path.join(self._data_dir, f"depth{self._frameNo:05}_{viewNum:05}.png")
        output_depth = crop_center_and_resize(depth)
        # output_depth = np.array(output_depth * 1000., dtype=np.uint16)  # [m] -> [mm]
        output_depth = np.array((1.0 - output_depth) * 50000., dtype=np.uint16)
        cv2.imwrite(depth_path, output_depth)

        pd.to_pickle(
            self._camera.get_world_pose(),
            os.path.join(self._data_dir, f"camera_info{self._frameNo:05}_{viewNum:05}.pkl"),
        )

    # def save_image(self, viewNum=None):
    #     rgb = self._camera.get_current_frame()["rgba"][:, :, :3]
    #     cv2.imwrite(
    #         os.path.join(self._data_dir, f"rgb{self._frameNo:05}_{viewNum:05}.jpg"),
    #         cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    #     )
    #     pd.to_pickle(
    #         self._camera.get_world_pose(),
    #         os.path.join(self._data_dir, f"camera_info{self._frameNo:05}_{viewNum:05}.pkl"),
    #     )

    def incrementFrameNumber(self):
        self._frameNo += 1

    def convert_to_force_distribution(self, contact_positions, impulse_values, log_scale=False):
        fmap = forcemap.GridForceMap("konbini_shelf")
        d = fmap.getDensity(contact_positions, impulse_values)
        if log_scale:
            d = np.log(1 + d)
        return d


class DatasetGenerator(metaclass=ABCMeta):
    def __init__(self, scene, output_smoothed_force=False):
        self._scene = scene
        ec = self._scene._env_config
        self._recorder = Recorder(self._scene._camera,
                                  crop_size=ec['crop_size'],
                                  output_image_size=ec['output_image_size'],
                                  output_smoothed_force=output_smoothed_force)

    def create(self, number_of_scenes, number_of_views=5):
        self._scene._world.initialize_physics()
        self._scene._world.play()

        for frameNum in range(number_of_scenes):
            self._scene.change_scene()
            self._scene.wait_for_stability()

            for viewNum in range(number_of_views):
                self._scene.change_observation_condition()
                self._scene.wait_for_stability(count=10)
                self._recorder.save_image(viewNum)
            self._recorder.save_state(self._scene)
            self._recorder.incrementFrameNumber()

        while simulation_app.is_running():
            self._scene._world.step(render=True)

        self._scene._world.stop()
        simulation_app.close()


from pathlib import Path

class ViewChanger:
    def __init__(self, scene, task_name="tabletop240304"):
        self._scene = scene
        self._task_name = task_name
        self._root_dir = Path(os.path.expanduser("~")) / "Dataset/forcemap"
        self._viewpoint = None  # Viewpoint for AIREC demo (2024.09)
        self._n_views = 3

    def load_bin_state(self, idx):
        p = self._root_dir / self._task_name / f"bin_state{idx:05d}.pkl"
        return pd.read_pickle(p)

    def get_primitive(self, name):
        o = [o for o in self._scene._loaded_objects if o.get_name() == name][0]
        return o.get_primitive()

    def restore_scene(self, bs):
        for name, pose in bs:
            print(name)
            prim = self.get_primitive(name)
            position = pose[0]
            rot_vec = rot_utils.quats_to_rotvecs(pose[1])
            angle = np.linalg.norm(rot_vec)
            axis = rot_vec / angle
            set_pose(prim, (position, (axis, angle)))

    def update_viewpoint(self, vp):
        self._scene.change_observation_condition()

    def save_scene_image(self):
        pass

    def create(self):
        self._scene._world.initialize_physics()
        self._scene._world.play()

        for idx in range(3):
            bs = self.load_bin_state(idx)
            self.restore_scene(bs)
            self._scene.wait_for_stability()

            for n in range(self._n_views):
                self.update_viewpoint(self._viewpoint)
                self._scene.wait_for_stability(count=10)
                self.save_scene_image()

        while simulation_app.is_running():
            self._scene._world.step(render=True)

        self._scene._world.stop()
        simulation_app.close()


# Konbini objects
# usd_files = glob.glob(os.path.join(os.environ['HOME'], '/Program/moonshot/ae_lstm/specification/meshes/objects/ycb/usd/*/*.usd'))
# names = [os.path.splitext(usd_file.split("/")[-1])[0] for usd_file in usd_files]


# Seria basket scene (IROS2023, moonshot interim demo.)
# scene = RandomSeriaBasketScene(world, conf)


# Basket scene for demo
# env_usd_file = f'{os.environ["HOME"]}/Dataset/scenes/ycb_piled_scene.usd'
# conf = ObjectInfo("ycb_conveni_v1_small")
# scene = RandomSeriaBasketScene(world, conf)



