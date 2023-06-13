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

from pxr import Gf, Usd, UsdGeom
import omni.usd

import numpy as np
import os, glob, cv2, scipy
import pandas as pd
import matplotlib.pyplot as plt

# from omni.isaac.dynamic_control import _dynamic_control
# dc = _dynamic_control.acquire_dynamic_control_interface()

from force_estimation import forcemap


# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

world = World(stage_units_in_meters=1.0)

# world.scene.add_default_ground_plane()

asset_path = os.environ["HOME"] + "/Downloads/Collected_ycb_piled_scene/simple_shelf_scene.usd"
usd_files = glob.glob(os.path.join(os.environ['HOME'], 'Dataset/Konbini/VER002/Seamless/vt2048/*/*/*.usd'))


simulation_context = SimulationContext()
# add_reference_to_stage(usd_path=asset_path, prim_path="/World")


masses = {
    '15_TOPPO-ART-pl2048SA': 0.072,
    '05_JIF-ART-pl2048SA': 0.358,
    # 'akaikitsune_mini-ART-pl2048SA': 0.136, # mesh is too bad
    '21_MT-KINOKO-ART-pl2048SA': 0.074,
    '20_MELTYKISS-ART-pl2048SA': 0.050, # not found in csv
    '07_GREEN-TEA-ART-pl2048SA': 0.040,
    '14_SHAMPOO-ART-pl2048SA': 0.633,
    'face_tawel-ART-pl2048SA': 0.179,
    '18_WAKAME-SOUP-ART-pl2048SA': 0.050,
    '16_CHOCO-RUSK-ART-pl2048SA': 0.090,
    '17_BUTTER-COOKIE-ART-pl2048SA': 0.120,
    '16cha_660-ART-pl2048SA': 0.656,
    '7i_barley_tea-ART-pl2048SA': 1.508,
    '2nd_cupnoodle_origin-ART-pl2048SA': 0.077,
    '11_XYLITOL-ART-pl2048SA': 0.143,
    '28_KOALAS-MARCH-ART-pl2048SA': 0.050,
    '19_POCKY-ART-pl2048SA': 0.072,
    '7i_edamamearare-ART-pl2048SA': 0.040,
    '13_CLORETS-ART-pl2048SA': 0.140,
    'vc_3000_dozen-ART-pl2048SA': 0.090,
    'Ayataka_pet1l-ART-pl2048SA': 1.006,
}


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
    properties = prim.GetPropertyNames()
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


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location=[0, 0, 0.1]):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")
    envPrim.GetReferences().AddReference(prim_usd_path)
    set_translate(envPrim, Gf.Vec3d(*location))
    prim = stage.GetPrimAtPath(envPrim.GetPath().pathString)
    return prim


def create_camera_D415():
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


class Object:
    def __init__(self, usd_file, object_id):
        self._usd_file = usd_file
        self._prim = create_prim_from_usd(world.stage, f'/World/object{object_id}', usd_file)
        self._id = object_id
        self._name = os.path.splitext(usd_file.split("/")[-1])[0]
        utils.setRigidBody(self._prim, "convexDecomposition", False)
        for node_prim in Usd.PrimRange(self._prim):
            for prim in Usd.PrimRange(node_prim):
                if prim.IsA(UsdGeom.Mesh):
                    print('SET_MASS: ', self._name, set_mass(prim, kg=masses[self._name]))  # why is this called 3 times?
        self.attach_contact_sensor()

    def get_primitive(self):
        return self._prim

    def get_name(self):
        return self._name

    def get_ID(self):
        return self._id

    def get_contact_sensor(self):
        return self._contact_sensor

    def attach_contact_sensor(self):
        self._contact_sensor = world.scene.add(
                                    ContactSensor(
                                        prim_path="/World/object{}/contact_sensor".format(self._id),
                                        name=f'{self._id}_cs',
                                        min_threshold=0,
                                        max_threshold=10000000,
                                        radius=0.5,
                                        translation=np.array([0, 0, 0])
                                    ))


loaded_objects = []
used_objects = []


def reset_object_positions():
    global used_objects
    for i, o in enumerate(loaded_objects):
        prim = o.get_primitive()
        set_translate(prim, Gf.Vec3d(0.1*i, 0, 0))
    used_objects = []


def place_objects(n):
    global used_objects
    used_objects = []
    for i, o in enumerate(np.random.choice(loaded_objects, n, replace=False)):
        prim = o.get_primitive()
        pos_xy = np.array([-0.10, 0.0]) + np.array([0.2, 0.3]) * (np.random.random(2) - 0.5)
        pos_z = 0.75 + 0.03 * i

        theta = 180 * np.random.random()
        phi = 360 * np.random.random()
        axis = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)]
        angle = 360 * np.random.random()
        set_pose(prim, ([pos_xy[0], pos_xy[1], pos_z], (axis, angle)))
        used_objects.append(o)


def wait_for_stability(count=100):
    stable = True
    for n in range(count):
        world.step(render=True)
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


# prim = world.stage.GetPrimAtPath('/World')
# stage = get_current_stage()
# scope = UsdGeom.Scope.Define(world.stage, 'Objects')
# prim.GetReferences().AddReference(asset_path)


def create_env():
    create_prim_from_usd(world.stage, '/World/env', asset_path, Gf.Vec3d([0, 0, 0.0]))


def create_objects(number_of_samples=30):
    global loaded_objects
    for i, usd_file in enumerate(usd_files[:number_of_samples]):
        loaded_objects.append(Object(usd_file, i))


def set_mass(prim, kg):
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(kg)


# def delete_objects():
#     global loaded_objects, contact_sensors
#     for i, _ in enumerate(loaded_objects):
#         # world.scene.remove_object(f'{i}_cs')
#         world.scene.clear()
#         prims.delete_prim(f'/World/object{i}')
#     loaded_objects = []
#     contact_sensors = []


number_of_lights = 3


def create_lights():
    for i in range(number_of_lights):
        prim_utils.create_prim(
            f'/World/Light/SphereLight{i}',
            "SphereLight",
            translation=(0.0, 0.0, 3.0),
            attributes={"radius": 1.0, "intensity": 5000.0, "color": (0.8, 0.8, 0.8)},
        )


def randomize_lights():
    for i in range(number_of_lights):
        prim = world.stage.GetPrimAtPath(f'/World/Light/SphereLight{i}')
        pos_xy = np.array([-2.5, -2.5]) + 5.0 * np.random.random(2)
        pos_z = 2.5 + 1.5 * np.random.random()
        set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))
        prim.GetAttribute("intensity").Set(2000. + 8000. * np.random.random())
        prim.GetAttribute("color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))
        # prim.GetAttribute("scale").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))        


def randomize_camera_parameters():
    global camera
    position = np.array([0.5, 0.0, 1.25]) + 0.04 * (np.random.random(3) - 0.5)
    # position = np.array([1.2, 0.0, 2.0]) + 0.04 * (np.random.random(3) - 0.5)
    orientation = rot_utils.euler_angles_to_quats(np.array([0, 40, 180] + 6 * (np.random.random(3) - 0.5)), degrees=True)
    camera.set_world_pose(position, orientation)
    # camera.set_focal_length(1.88)
    # camera.set_focus_distance(40)
    # camera.set_horizontal_aperture(2.7288)
    # camera.set_vertical_aperture(1.5498)


def randomize_object_colors():
    # for o in used_objects:
    #     object_id = o.get_ID()
    #     prim = world.stage.GetPrimAtPath(f'/World/object{object_id}/Looks/material_0')
    #     prim.GetAttribute("color tint").Set()
    pass


create_env()
create_lights()
camera = create_camera_D415()
camera.initialize()


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


def get_bin_state():
    prim_path = "/World/_09_gelatin_box/contact_sensor"
    obj = world.scene.get_object(prim_path)
    return obj.get_world_pose()


def convert_to_force_distribution(contact_positions, impulse_values, bin_state, log_scale=True):
    fmap = forcemap.GridForceMap('konbini_shelf')
    d = fmap.getDensity(contact_positions, impulse_values)
    if log_scale:
        d = np.log(1 + d)
    return d


# sr = world.scene._scene_registry
# print(sr._rigid_objects, sr._geometry_objects)

# create_objects()
# reset_object_positions()

def create_toppos(n=20):
    global loaded_objects
    loaded_objects = []
    usd_file = os.path.join(os.environ['HOME'], 'Dataset/Konbini/VER002/Seamless/vt2048/15_TOPPO-ART-vt2048SA/15_TOPPO-ART-pl2048SA/15_TOPPO-ART-pl2048SA.usd')
    for i in range(n):
        loaded_objects.append(Object(usd_file, i))


def create_toppo_scene(n=20):
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
    randomize_object_colors()
    wait_for_stability(count=1000)


create_toppos()


# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
simulation_context.play()


def save(frameNo, rgb, bin_state, contact_raw_data, force_distribution, camera_pose):
    data_dir = 'data/'
    cv2.imwrite(os.path.join(data_dir, 'rgb{:05d}.jpg'.format(frameNo)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    pd.to_pickle(bin_state, os.path.join(data_dir, 'bin_state{:05d}.pkl'.format(frameNo)))
    pd.to_pickle(contact_raw_data, os.path.join(data_dir, 'contact_raw_data{:05d}.pkl'.format(frameNo)))
    if force_distribution != None:
        pd.to_pickle(force_distribution, os.path.join(data_dir, 'force_zip{:05d}.pkl'.format(frameNo)))
    pd.to_pickle(camera_pose, os.path.join(data_dir, 'camera_info{:05d}.pkl'.format(frameNo)))


def create_random_scene():
    world.reset()
    place_objects(10)
    randomize_lights()
    randomize_camera_parameters()
    randomize_object_colors()
    wait_for_stability()


def record_scene(frameNo, output_distribution=False):
    rgb = camera.get_current_frame()['rgba'][:, :, :3]

    contact_positions = []
    impulse_values = []
    bin_state = []

    for o in used_objects:
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
        print(f'SCENE[{frameNo},{o.get_name()}]:', trans, quat)

    if output_distribution:
        force_dist = convert_to_force_distribution(contact_positions, impulse_values, bin_state)
    else:
        force_dist = None
    save(frameNo, rgb, bin_state, (contact_positions, impulse_values), force_dist, camera.get_world_pose())


def create_random_dataset(number_of_scenes):
    for frameNo in range(number_of_scenes):
        create_random_scene()
        record_scene(frameNo)
        # simulation_context.stop()
        # if world.current_time_step_index == 0:

    while simulation_app.is_running():
        world.step(render=True)

    # simulation_context.step(render=True)
    simulation_context.stop()
    simulation_app.close()


def create_toppo_data():
    create_toppo_scene()
    record_scene(0)

    while simulation_app.is_running():
        world.step(render=True)

    # simulation_context.step(render=True)
    simulation_context.stop()
    simulation_app.close()


# create_random_dataset(number_of_scenes=5)
create_toppo_data()
