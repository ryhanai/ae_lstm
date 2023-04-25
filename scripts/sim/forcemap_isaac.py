
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor
from omni.isaac.sensor import Camera
from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.stage import add_reference_to_stage
# from omni.isaac.core.utils import prims
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import os, glob, cv2, scipy
import pandas as pd

import omni
from pxr import UsdPhysics
from omni.physx.scripts import utils
import omni.isaac.core.utils.prims as prim_utils
import matplotlib.pyplot as plt

from pxr import Gf, Usd, UsdGeom
# import carb
import omni.usd

# from omni.isaac.dynamic_control import _dynamic_control
# dc = _dynamic_control.acquire_dynamic_control_interface()


# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

world = World(stage_units_in_meters=1.0)

# world.scene.add_default_ground_plane()

asset_path = os.environ["HOME"] + "/Downloads/Collected_ycb_piled_scene/simple_shelf_scene.usd"

simulation_context = SimulationContext()
# add_reference_to_stage(usd_path=asset_path, prim_path="/World")


usd_files = glob.glob('/home/ryo/Dataset/Konbini/VER002/Seamless/vt2048/*/*/*.usd')


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


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location=[0, 0, 0.1], set_rigid_body=True):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")
    envPrim.GetReferences().AddReference(prim_usd_path)
    set_translate(envPrim, Gf.Vec3d(*location))
    prim = stage.GetPrimAtPath(envPrim.GetPath().pathString)
    if set_rigid_body:
        utils.setRigidBody(prim, "convexDecomposition", False)
        set_mass(prim, kg=0.3)
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


# def set_world_pose(prim, position=[0, 0, 0]):
#     xform = UsdGeom.Xformable(prim)
#     xform.ClearXformOpOrder() # Is this necessary?
#     transform = xform.AddTransformOp()
#     mat = Gf.Matrix4d()
#     mat.SetTranslateOnly(Gf.Vec3d(*position))
#     mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), 0))
#     transform.Set(mat)


class Object:
    def __init__(self, usd_file, object_id):
        self._usd_file = usd_file
        self._prim = create_prim_from_usd(world.stage, f'/World/object{object_id}', usd_file)
        self._id = object_id
        self._name = os.path.splitext(usd_file.split("/")[-1])[0]
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
        set_translate(prim, Gf.Vec3d(0.05*i, 0, 0))
    used_objects = []


def place_objects(n):
    global used_objects
    for i, o in enumerate(np.random.choice(loaded_objects, n)):
        prim = o.get_primitive()
        pos_xy = np.array([-0.15, -0.15]) + 0.3 * np.random.random(2)
        pos_z = 0.74 + 0.01 * i
        set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))
        used_objects.append(o)


def wait_for_stability():
    stable = True
    for n in range(150):
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
# ch_prim = world.stage.GetPrimAtPath('/World/simple_shelf')


def create_env():
    create_prim_from_usd(world.stage, '/World/env', asset_path, Gf.Vec3d([0, 0, 0.0]), set_rigid_body=False)


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
        pos_z = 2.0 + 2.0 * np.random.random()
        set_translate(prim, Gf.Vec3d(pos_xy[0], pos_xy[1], pos_z))
        prim.GetAttribute("intensity").Set(2000. + 18000. * np.random.random())
        prim.GetAttribute("color").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))
        # prim.GetAttribute("scale").Set(Gf.Vec3f(*(np.array([0.2, 0.2, 0.2]) + 0.6 * np.random.random(3))))        


def randomize_camera_parameters():
    global camera
    position = np.array([0.4, 0.0, 1.15]) + 0.04 * (np.random.random(3) - 0.5)
    orientation = rot_utils.euler_angles_to_quats(np.array([0, 45, 180] + 8 * (np.random.random(3) - 0.5)), degrees=True)
    camera.set_world_pose(position, orientation)
    # camera.set_focal_length(1.88)
    # camera.set_focus_distance(40)
    # camera.set_horizontal_aperture(2.7288)
    # camera.set_vertical_aperture(1.5498)


def randomize_object_colors():
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


# def drop_an_object(name='_09_gelatin_box'):
#     prim = object_prims[name]
#     set_world_pose(prim)

# stage = omni.usd.get_context().get_stage()
# object_prims = {}
# for o in objects:
#     object_prims[o] = stage.GetPrimAtPath('/World/{}'.format(o))


# prim = object_prims[objects[0]]
# set_world_pose(prim)


# sr = world.scene._scene_registry
# print(sr._rigid_objects, sr._geometry_objects)

number_of_scenes = 3
create_objects()
reset_object_positions()


# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
simulation_context.play()


def save(frameNo, rgb, forcemap, bin_state, filtered_cs_data, camera_pose):
    data_dir = 'data/'
    cv2.imwrite(os.path.join(data_dir, 'rgb{:05d}.jpg'.format(frameNo)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    pd.to_pickle(bin_state, os.path.join(data_dir, 'bin_state{:05d}.pkl'.format(frameNo)))

    # filtered_contacts = list(filter(lambda x: scipy.linalg.norm(x['impulse']) > 1e-8, contacts))
    # contact_positions = np.array([x['position'] for x in filtered_contacts])
    # impulse_value = np.array([scipy.linalg.norm(x['impulse']) for x in filtered_contacts])

    # return contact_positions, impulse_value, bin_state


for frameNo in range(number_of_scenes):
    world.reset()
    place_objects(10)
    randomize_lights()
    randomize_camera_parameters()
    randomize_object_colors()

    # if world.is_playing():
    wait_for_stability()

    rgb = camera.get_current_frame()['rgba'][:, :, :3]

    cs_data = []
    bin_state = []
    for o in used_objects:
        contact_sensor = o.get_contact_sensor()
        cs_data.append(read_contact_sensor(contact_sensor))
        # print(contact_sensor.get_current_frame())
        # print(contact_sensor_contact_sensor_interface.get_contact_sensor_raw_data(contact_sensor.prim_path))

        prim = o.get_primitive()
        pose = omni.usd.utils.get_world_transform_matrix(prim)
        trans = pose.ExtractTranslation()
        quat = pose.ExtractRotation().GetQuaternion()
        bin_state.append((o.get_name(), (trans, quat)))
    
    camera_pose = camera.get_world_pose()
    forcemap = None
    
    filtered_cs_data = cs_data
    print(f'SAVE SCENE[{frameNo},{o.get_name()}]:', trans, quat)
    save(frameNo, rgb, forcemap, bin_state, filtered_cs_data, camera_pose)

    # simulation_context.stop()
    # if world.current_time_step_index == 0:


while simulation_app.is_running():
    world.step(render=True)

#     simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
