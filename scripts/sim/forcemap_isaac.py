# -*- coding: utf-8 -*-

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor
from omni.isaac.sensor import Camera
from omni.isaac.core import SimulationContext
import omni.isaac.core.utils.numpy.rotations as rot_utils

import omni
from pxr import UsdPhysics
from omni.physx.scripts import utils
import omni.isaac.core.utils.prims as prim_utils

from pxr import Gf, Usd, UsdGeom, UsdShade, Sdf
import omni.usd

import numpy as np
import os, glob, cv2, scipy
import pandas as pd
import matplotlib.pyplot as plt

# from omni.isaac.dynamic_control import _dynamic_control
# dc = _dynamic_control.acquire_dynamic_control_interface()

from force_estimation import forcemap
from abc import ABCMeta, abstractmethod


# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
# world.scene.add_default_ground_plane()


# YCB objects
usd_files = glob.glob('/home/ryo/Program/moonshot/ae_lstm/specification/meshes/objects/ycb/*/google_16k/textured/*.usd')
conf = {
    'asset_path' : os.environ["HOME"] + "/Downloads/Collected_ycb_piled_scene/ycb_piled_scene.usd",
    # 'asset_path' : os.environ["HOME"] + "/Downloads/green_table_scene.usd",    
    'names' : [usd_file.split('/')[-4] for usd_file in usd_files],
    'usd_files' : usd_files,
    'masses' : {
        '004_sugar_box': 0.514,
        '005_tomato_soup_can': 0.349,
        # '006_mustard_bottle': 0.603,
        '007_tuna_fish_can': 0.171,
        '008_pudding_box': 0.187,
        '009_gelatin_box': 0.097,
        '010_potted_meat_can': 0.370,
        '011_banana': 0.066,
        '012_strawberry': 0.018,
        '013_apple': 0.068,
        '014_lemon': 0.029,
        '016_pear': 0.049,
        '017_orange': 0.047,
        # '023_wine_glass': 0.133, # no 3D model
        '026_sponge': 0.0062,
        '040_large_marker': 0.0158,
        # '041_small_marker': 0.0082,
        '055_baseball': 0.148,
        '056_tennis_ball': 0.056,
        '061_foam_brick': 0.059,
        # additional objects
        # '001_chips_can' : 0.205,
        '002_master_chef_can' : 0.414,
        '015_peach' : 0.033,
        '018_plum' : 0.025,
        '021_bleach_cleanser' : 1.131,
        # '022_windex_bottle' : 1.022,
        '023_wine_glass' : 0.133,
        '024_bowl' : 0.147,
        '025_mug' : 0.118,
        '029_plate' : 0.279,
        '030_fork' : 0.034,
        '031_spoon' : 0.030,
        '032_knife' : 0.031,
        '033_spatula' : 0.0515,
        '035_power_drill' : 0.895,
        '036_wood_block' : 0.729,
        '037_scissors' : 0.082,
        # '041_small_marker' : 0.0082,
        '042_adjustable_wrench' : 0.252,
        '043_phillips_screwdriver' : 0.097,
        '044_flat_screwdriver' : 0.0984,
        '048_hammer' : 0.665,
        # '049_small_clamp' : 0.0192,
        '050_medium_clamp' : 0.059,
        '051_large_clamp' : 0.125,
        '052_extra_large_clamp' : 0.202,
        '053_mini_soccer_ball' : 0.123,
        '054_softball' : 0.191,
        '057_racquetball' : 0.041,
        '058_golf_ball' : 0.665,
        '077_rubiks_cube' : 0.252,
    },
    'center_of_objects' : {
        '004_sugar_box': [-0.01, -0.018, 0.09],
        '005_tomato_soup_can': [-0.01, 0.085, 0.054],
        '007_tuna_fish_can': [-0.026, -0.021, 0.015],
        '008_pudding_box': [0.004, 0.02, 0.02],
        '009_gelatin_box': [-0.022, -0.007, 0.014],
        '010_potted_meat_can': [-0.035, -0.026, 0.04],
        '011_banana': [-0.013, 0.013, 0.019],
        '012_strawberry': [-0.001, 0.017, 0.022],
        '013_apple': [0.001, -0.001, 0.04],
        '014_lemon': [-0.011, 0.024, 0.027],
        '016_pear': [-0.015, 0.009, 0.03],
        '017_orange': [-0.007, -0.015, 0.038],
        '026_sponge': [-0.015, 0.019, 0.01],
        '040_large_marker': [-0.036, -0.002, 0.01],
        '055_baseball': [-0.01, -0.045, 0.04],
        '056_tennis_ball': [0.008, -0.041, 0.035],
        '061_foam_brick': [-0.018, 0.019, 0.025],
    }
}


# asset_path = os.environ["HOME"] + "/Downloads/Collected_ycb_piled_scene/simple_shelf_scene.usd"


# add_reference_to_stage(usd_path=asset_path, prim_path="/World")
# prim = world.stage.GetPrimAtPath('/World')
# stage = get_current_stage()
# scope = UsdGeom.Scope.Define(world.stage, 'Objects')
# prim.GetReferences().AddReference(asset_path)

world = World(stage_units_in_meters=1.0)
simulation_context = SimulationContext()

# masses = {
#     '15_TOPPO-ART-pl2048SA': 0.072,
#     '05_JIF-ART-pl2048SA': 0.358,
#     # 'akaikitsune_mini-ART-pl2048SA': 0.136, # mesh is too bad
#     '21_MT-KINOKO-ART-pl2048SA': 0.074,
#     '20_MELTYKISS-ART-pl2048SA': 0.050, # not found in csv
#     '07_GREEN-TEA-ART-pl2048SA': 0.040,
#     '14_SHAMPOO-ART-pl2048SA': 0.633,
#     'face_tawel-ART-pl2048SA': 0.179,
#     '18_WAKAME-SOUP-ART-pl2048SA': 0.050,
#     '16_CHOCO-RUSK-ART-pl2048SA': 0.090,
#     '17_BUTTER-COOKIE-ART-pl2048SA': 0.120,
#     '16cha_660-ART-pl2048SA': 0.656,
#     '7i_barley_tea-ART-pl2048SA': 1.508,
#     '2nd_cupnoodle_origin-ART-pl2048SA': 0.077,
#     '11_XYLITOL-ART-pl2048SA': 0.143,
#     '28_KOALAS-MARCH-ART-pl2048SA': 0.050,
#     '19_POCKY-ART-pl2048SA': 0.072,
#     '7i_edamamearare-ART-pl2048SA': 0.040,
#     '13_CLORETS-ART-pl2048SA': 0.140,
#     'vc_3000_dozen-ART-pl2048SA': 0.090,
#     'Ayataka_pet1l-ART-pl2048SA': 1.006,
# }


def set_pose(prim, pose):
    p, (axis, angle) = pose
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder() # Is this necessary?
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
    # properties = prim.GetPropertyNames()
    # if "xformOp:rotate" in properties:
    #     rotate_attr = prim.GetAttribute("xformOp:rotate")
    #     rotate_attr.Set(rot_mat)
    # elif "xformOp:transform" in properties:
    #     transform_attr = prim.GetAttribute("xformOp:transform")
    #     matrix = prim.GetAttribute("xformOp:transform").Get()
    #     matrix.SetRotateOnly(rot_mat.ExtractRotation())
    #     transform_attr.Set(matrix)
    # else:
    #     xform = UsdGeom.Xformable(prim)
    #     xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
    #     xform_op.Set(Gf.Matrix4d().SetRotate(rot_mat))
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


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location=[0, 0, 0.1]):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")
    envPrim.GetReferences().AddReference(prim_usd_path)
    set_translate(envPrim, Gf.Vec3d(*location))
    prim = stage.GetPrimAtPath(envPrim.GetPath().pathString)
    return prim


class Object:
    def __init__(self, name, usd_file, object_id, mass, world):
        self._usd_file = usd_file
        self._prim = create_prim_from_usd(world.stage, f'/World/object{object_id}', usd_file)
        self._id = object_id
        self._name = name
        utils.setRigidBody(self._prim, "convexDecomposition", False)
        for node_prim in Usd.PrimRange(self._prim):
            for prim in Usd.PrimRange(node_prim):
                if prim.IsA(UsdGeom.Mesh):
                    print('SET_MASS: ', self._name, set_mass(prim, kg=mass))  # why is this called 3 times?
        self.attach_contact_sensor(world)

    def get_primitive(self):
        return self._prim

    def get_name(self):
        return self._name

    def get_ID(self):
        return self._id

    def get_contact_sensor(self):
        return self._contact_sensor

    def attach_contact_sensor(self, world):
        self._contact_sensor = world.scene.add(
                                    ContactSensor(
                                        prim_path="/World/object{}/contact_sensor".format(self._id),
                                        name=f'{self._id}_cs',
                                        min_threshold=0,
                                        max_threshold=10000000,
                                        radius=0.5,
                                        translation=np.array([0, 0, 0])
                                    ))


class Scene:
    def __init__(self, world):
        self._world = world
        self._loaded_objects = []
        self._used_objects = []
        self._number_of_lights = 3

        self.create_env()
        self.create_lights()
        self._camera = self.create_camera_D415()
        self._camera.initialize()
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

    def get_active_objects(self):
        return self._used_objects

    def sample_object_pose(self):
        pos_xy = np.array([0.10, 0.0]) + np.array([0.2, 0.3]) * (np.random.random(2) - 0.5)
        pos_z = 0.76 + 0.3 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [pos_xy[0], pos_xy[1], pos_z], (axis, angle)

    def place_objects(self, n):
        self._used_objects = []
        for i, o in enumerate(np.random.choice(self._loaded_objects, n, replace=False)):
            print(f'placing {o.get_name()}')
            prim = o.get_primitive()
            contact_sensor = o.get_contact_sensor()

            while True:
                pose = self.sample_object_pose()
                set_pose(prim, pose)

                no_collision = True
                for j in range(3):
                    self.wait_for_stability(count=1)
                    contacts = read_contact_sensor(contact_sensor)
                    if len(contacts) > 0:
                        # if c['body1'] != '/World/env/simple_shelf':
                        # print(f"initial collision with {c['body1']}. replace ...")
                        no_collision = False
                if no_collision:
                    break

            print(f'{o.get_name()} placed')
            self._used_objects.append(o)

    def wait_for_stability(self, count=100):
        stable = True
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
        create_prim_from_usd(self._world.stage, '/World/env', conf['asset_path'], Gf.Vec3d([0, 0, 0.0]))

    def create_camera_D415(self):
        camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.4, 0.0, 1.15]),
            frequency=20,
            resolution=(1280, 720),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 45, 180]), degrees=True))
        camera.set_focal_length(1.88)
        camera.set_focus_distance(40)
        camera.set_horizontal_aperture(2.7288)
        camera.set_vertical_aperture(1.5498)
        camera.set_clipping_range(0.1, 5.0)
        return camera

    def create_lights(self):
        for i in range(self._number_of_lights):
            prim_utils.create_prim(
                f'/World/Light/SphereLight{i}',
                "SphereLight",
                translation=(0.0, 0.0, 3.0),
                attributes={"radius": 1.0, "intensity": 5000.0, "color": (0.8, 0.8, 0.8)},
            )

    def randomize_lights(self):
        for i in range(self._number_of_lights):
            prim = self._world.stage.GetPrimAtPath(f'/World/Light/SphereLight{i}')
            pos_xy = np.array([-2.5, -2.5]) + 5.0 * np.random.random(2)
            pos_z = 2.5 + 1.5 * np.random.random()
            set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))
            prim.GetAttribute("intensity").Set(2000. + 8000. * np.random.random())
            prim.GetAttribute("color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))
            # prim.GetAttribute("scale").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))        

    def randomize_camera_parameters(self):
        d = 0.7 + 0.2 * (np.random.random() - 0.5)
        theta = 20 + 50 * np.random.random()
        phi = -50 + 100 * np.random.random()
        th = np.deg2rad(theta)
        ph = np.deg2rad(phi)
        position = np.array([0, 0, 0.75]) + d * np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th)])
        theta2 = theta + 6 * (np.random.random() - 0.5)
        phi2 = phi + 6 * (np.random.random() - 0.5)
        orientation = rot_utils.euler_angles_to_quats(np.array([0, theta2, 180+phi2]), degrees=True)
        print(f'position={position}, orientation={orientation}')
        self._camera.set_world_pose(position, orientation)

        self._camera.set_focal_length(1.88 + 0.1128 * (np.random.random() - 0.5))  # 3% of the spec
        self._camera.set_focus_distance(50)
        self._camera.set_horizontal_aperture(2.6034 + 0.1562 * (np.random.random() - 0.5))
        self._camera.set_vertical_aperture(1.4621 + 0.0877 * (np.random.random() - 0.5))

    def randomize_object_colors(self):
        for o in self._used_objects:
            object_id = o.get_ID()
            mtl_prim = world.stage.GetPrimAtPath(f'/World/object{object_id}/Looks/material_0')
            # mtl_prim.GetAttribute("color tint").Set()
            omni.usd.create_material_input(mtl_prim, "diffuse_tint", Gf.Vec3f(*(0.5 + 0.5*np.random.random(3))), Sdf.ValueTypeNames.Color3f)
            mtl_shade = UsdShade.Material(mtl_prim)
            obj_prim = world.stage.GetPrimAtPath(f'/World/object{object_id}')
            UsdShade.MaterialBindingAPI(obj_prim).Bind(mtl_shade, UsdShade.Tokens.strongerThanDescendants)



class RandomScene(Scene):
    def __init__(self, world, conf):
        self._names = conf['names']
        self._usd_files = conf['usd_files']
        super().__init__(world)

    def create_objects(self):
        for i, (name, usd_file) in enumerate(zip(self._names, self._usd_files)):
            self._loaded_objects.append(Object(name, usd_file, i, conf['masses'][name], self._world))
        self.reset_object_positions()

    def reset_object_positions(self):
        for i, o in enumerate(self._loaded_objects):
            prim = o.get_primitive()
            set_translate(prim, Gf.Vec3d(0.1*i, 0, 0))
        self._used_objects = []

    def change_scene(self):
        self._world.reset()
        self.place_objects(10)

    def change_observation_condition(self):
        self.randomize_lights()
        self.randomize_camera_parameters()
        self.randomize_object_colors()


class RandomSeriaBasketScene(RandomScene):
    def __init__(self, world, conf):
        self._names = conf['names']
        self._usd_files = conf['usd_files']
        super().__init__(world, conf)

    def change_scene(self):
        self._world.reset()
        number_of_objects = np.clip(np.random.poisson(5), 1, 8)
        self.place_objects(number_of_objects)

    def sample_object_pose(self):
        xy = np.array([0.15, 0.08]) * (np.random.random(2) - 0.5)
        z = 0.75 + 0.25 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [xy[0], xy[1], z], (axis, angle)

    def randomize_camera_parameters(self):
        position = np.array([0, 0, 1.358]) + 0.02 * (np.random.random(3) - 0.5)
        orientation = rot_utils.euler_angles_to_quats(np.array([0, 90, 180] + 6 * (np.random.random(3) - 0.5)), degrees=True)
        self._camera.set_world_pose(position, orientation)


class RandomTableScene(RandomScene):
    def __init__(self, world, conf):
        self._names = conf['names']
        self._usd_files = conf['usd_files']
        super().__init__(world, conf)

    def change_scene(self):
        self._world.reset()
        number_of_objects = np.clip(np.random.poisson(7), 1, 10)
        self.place_objects(number_of_objects)

    def sample_object_pose(self):
        xy = np.array([0.15, 0.15]) * (np.random.random(2) - 0.5)
        z = 0.75 + 0.25 * np.random.random()
        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        return [xy[0], xy[1], z], (axis, angle)

    def randomize_camera_parameters(self):
        d = 0.7 + 0.2 * (np.random.random() - 0.5)
        theta0 = np.radians(20)
        theta1 = np.radians(60)
        z = np.random.random()
        th = np.arcsin(z * (np.sin(theta1) - np.sin(theta0)) + np.sin(theta0))
        phi = 360 * np.random.random()
        ph = np.deg2rad(phi)
        position = np.array([0, 0, 0.75]) + d * np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th)])
        theta2 = np.degrees(th) + 6 * (np.random.random() - 0.5)
        phi2 = phi + 6 * (np.random.random() - 0.5)
        orientation = rot_utils.euler_angles_to_quats(np.array([0, theta2, 180+phi2]), degrees=True)
        print(f'position={position}, orientation={orientation}')
        self._camera.set_world_pose(position, orientation)

        self._camera.set_focal_length(1.88 + 0.1128 * (np.random.random() - 0.5))  # 3% of the spec
        self._camera.set_focus_distance(50)
        self._camera.set_horizontal_aperture(2.6034 + 0.1562 * (np.random.random() - 0.5))
        self._camera.set_vertical_aperture(1.4621 + 0.0877 * (np.random.random() - 0.5))


# def delete_objects():
#     global loaded_objects, contact_sensors
#     for i, _ in enumerate(loaded_objects):
#         # world.scene.remove_object(f'{i}_cs')
#         world.scene.clear()
#         prims.delete_prim(f'/World/object{i}')
#     loaded_objects = []
#     contact_sensors = []


def read_contact_sensor(contact_sensor):
    csif = contact_sensor._contact_sensor_interface
    cs_raw_data = csif.get_contact_sensor_raw_data(contact_sensor.prim_path)
    backend_utils = simulation_context.backend_utils
    device = simulation_context.device

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
                                    [
                                        cs_raw_data["normal"][i][0],
                                        cs_raw_data["normal"][i][1],
                                        cs_raw_data["normal"][i][2]
                                    ],
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


# sr = world.scene._scene_registry
# print(sr._rigid_objects, sr._geometry_objects)


class AlignedToppoScene(Scene):
    def __init__(self):
        self.create_toppos(20)

    def create_toppos(self, n):
        self._loaded_objects = []
        usd_file = os.path.join(os.environ['HOME'], 'Dataset/Konbini/VER002/Seamless/vt2048/15_TOPPO-ART-vt2048SA/15_TOPPO-ART-pl2048SA/15_TOPPO-ART-pl2048SA.usd')
        for i in range(n):
            self._loaded_objects.append(Object(usd_file, i, self._world))

    def create_toppo_scene(self, n):
        global used_objects
        used_objects = []
        
        world.reset()        
        for i, o in enumerate(loaded_objects):
            prim = o.get_primitive()
            pos_xy = np.array([0.0, -0.1]) + np.array([0.0, 0.025]) * i
            pos_z = 0.73

            axis = [1, 0, 0]
            angle = 90
            set_pose(prim, ([pos_xy[0], pos_xy[1], pos_z], (axis, angle)))
            used_objects.append(o)

        randomize_lights()
        randomize_camera_parameters()
        # randomize_object_colors()
        wait_for_stability(count=1200)


class Recorder:
    def __init__(self, camera, output_force=False):
        self.__output_force = output_force
        self._frameNo = 0
        self._camera = camera
        self._data_dir = 'data'

    def save_state(self, objects_to_monitor):
        """
            Record bin-state and force
        """
        contact_positions = []
        impulse_values = []
        bin_state = []

        for o in objects_to_monitor:
            contact_sensor = o.get_contact_sensor()
            contacts = read_contact_sensor(contact_sensor)
            # print(contact_sensor.get_current_frame())
            # print(contact_sensor_contact_sensor_interface.get_contact_sensor_raw_data(contact_sensor.prim_path))

            for contact in contacts:
                if scipy.linalg.norm(contact['impulse']) > 1e-8:
                    contact_positions.append(contact['position'])
                    impulse_values.append(scipy.linalg.norm(contact['impulse']))

            prim = o.get_primitive()
            pose = omni.usd.utils.get_world_transform_matrix(prim)
            trans = pose.ExtractTranslation()
            trans = np.array(trans)
            quat = pose.ExtractRotation().GetQuaternion()
            quat = np.array(list(quat.imaginary) + [quat.real])
            bin_state.append((o.get_name(), (trans, quat)))
            print(f'SCENE[{self._frameNo},{o.get_name()}]:', trans, quat)

        pd.to_pickle(bin_state, os.path.join(self._data_dir, 'bin_state{:05d}.pkl'.format(self._frameNo)))
        pd.to_pickle((contact_positions, impulse_values), os.path.join(self._data_dir, 'contact_raw_data{:05d}.pkl'.format(self._frameNo)))

        if self.__output_force:
            force_dist = self.convert_to_force_distribution(contact_positions, impulse_values)
            pd.to_pickle(force_dist, os.path.join(self._data_dir, 'force_zip{:05d}.pkl'.format(self._frameNo)))

    def save_image(self, viewNum=None):
        rgb = self._camera.get_current_frame()['rgba'][:, :, :3]
        cv2.imwrite(os.path.join(self._data_dir, f'rgb{self._frameNo:05}_{viewNum:05}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pd.to_pickle(self._camera.get_world_pose(), os.path.join(self._data_dir, f'camera_info{self._frameNo:05}_{viewNum:05}.pkl'))

    def incrementFrameNumber(self):
        self._frameNo += 1

    # def get_bin_state(self):
    #     prim_path = "/World/_09_gelatin_box/contact_sensor"
    #     obj = world.scene.get_object(prim_path)
    #     return obj.get_world_pose()

    def convert_to_force_distribution(self, contact_positions, impulse_values, log_scale=True):
        fmap = forcemap.GridForceMap('konbini_shelf')
        d = fmap.getDensity(contact_positions, impulse_values)
        if log_scale:
            d = np.log(1 + d)
        return d


class DatasetGenerator(metaclass=ABCMeta):
    def __init__(self, scene, output_force=True):
        self._scene = scene
        self._recorder = Recorder(self._scene._camera, output_force=output_force)

    def create(self, number_of_scenes):
        simulation_context.initialize_physics()
        simulation_context.play()

        for frameNum in range(number_of_scenes):
            self._scene.change_scene()
            self._scene.wait_for_stability()

            for viewNum in range(5):
                self._scene.change_observation_condition()
                self._scene.wait_for_stability(count=10)
                self._recorder.save_image(viewNum)
            self._recorder.save_state(self._scene.get_active_objects())
            self._recorder.incrementFrameNumber()

        # simulation_context.stop()
        # if world.current_time_step_index == 0:

        while simulation_app.is_running():
            self._scene._world.step(render=True)
        # simulation_context.step(render=True)
        simulation_context.stop()
        simulation_app.close()


# Konbini objects
# usd_files = glob.glob(os.path.join(os.environ['HOME'], '/Program/moonshot/ae_lstm/specification/meshes/objects/ycb/usd/*/*.usd'))
# names = [os.path.splitext(usd_file.split("/")[-1])[0] for usd_file in usd_files]

# Seria basket scene (IROS2023, moonshot interim demo.)
# scene = RandomSeriaBasketScene(world, conf)

scene = RandomTableScene(world, conf)
dataset = DatasetGenerator(scene, output_force=False)
dataset.create(10)
