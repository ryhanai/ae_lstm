
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import os
import omni
from omni.physx.scripts import utils
import glob

from pxr import Gf, Usd, UsdGeom
import carb
import omni.usd
import gc


# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

asset_path = os.environ["HOME"] + "/Downloads/Collected_ycb_piled_scene/simple_shelf_scene.usd"

simulation_context = SimulationContext()
# add_reference_to_stage(usd_path=asset_path, prim_path="/World")


# follow_sphere = my_world.scene.add(
#     VisualSphere(
#         name="follow_sphere", prim_path="/World/FollowSphere", radius=0.02, color=np.array([0.7, 0.0, 0.7])
#     )
# )


number_objects = 5
usd_files = glob.glob('/home/ryo/Dataset/Konbini/VER002/Seamless/vt2048/*/*/*.usd')[:number_objects]
objects = [os.path.splitext(s.split("/")[-1])[0] for s in usd_files]


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


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")
    envPrim.GetReferences().AddReference(prim_usd_path)
    set_translate(envPrim, location)
    return stage.GetPrimAtPath(envPrim.GetPath().pathString)


# prim = world.stage.GetPrimAtPath('/World')
# stage = get_current_stage()
# scope = UsdGeom.Scope.Define(world.stage, 'Objects')

# prim.GetReferences().AddReference(asset_path)
# ch_prim = world.stage.GetPrimAtPath('/World/simple_shelf')

create_prim_from_usd(world.stage, '/World/env', asset_path, Gf.Vec3d([0, 0, 0.0]))

for i, usd_file in enumerate(usd_files):
    pos = np.array([-0.1, -0.1, 0.8]) + 0.2 * np.random.random(3)
    prim = create_prim_from_usd(world.stage, '/World/object{}'.format(i), usd_file, Gf.Vec3d(list(pos)))
    utils.setRigidBody(prim, "convexDecomposition", False)

# contact_sensors = []
# for o in objects:
#     contact_sensors.append(
#         my_world.scene.add(
#             ContactSensor(
#                 prim_path="/World/{}/contact_sensor".format(o),
#                 name="{}_cs".format(o),
#                 min_threshold=0,
#                 max_threshold=10000000,
#                 radius=0.5,
#                 translation=np.array([0, 0, 0])
#             )
#         )
#     )


# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
# simulation_context.play()

# for i in range(1000):
#     simulation_context.step(render=True)

# contact_sensors[0].add_raw_contact_data_to_frame()


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
    obj = my_world.scene.get_object(prim_path)
    return obj.get_world_pose()


def drop_an_object(name='_09_gelatin_box'):
    prim = object_prims[name]
    set_world_pose(prim)


stage = omni.usd.get_context().get_stage()
object_prims = {}
for o in objects:
    object_prims[o] = stage.GetPrimAtPath('/World/{}'.format(o))


def set_world_pose(prim, position=Gf.Vec3d(0, 0, 1.5)):
    xform = UsdGeom.Xformable(prim)
    # xform.ClearXformOpOrder()
    transform = xform.AddTransformOp()
    mat = Gf.Matrix4d()
    mat.SetTranslateOnly(position)
    mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), 0))
    transform.Set(mat)


# prim = object_prims[objects[0]]
# set_world_pose(prim)


stable = False
cnt = 0
def is_scene_stable():
    global cnt
    if cnt > 500:
        cnt = 0
        return True
    else:
        cnt += 1
        return False


sr = world.scene._scene_registry
print(sr._rigid_objects, sr._geometry_objects)


while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():

        # while o in objects:
        #     if stable:
        #         stable = False
    
        #     else:
        #         if is_scene_stable():
        #             stable = True

            # print(contact_sensors[0].get_current_frame())
            # print(contact_sensors[0]._contact_sensor_interface.get_contact_sensor_raw_data(contact_sensors[0].prim_path))

#        print(read_contact_sensor(contact_sensors[0]))
        
        # prim = object_prims[objects[0]]
        # pose = omni.usd.utils.get_world_transform_matrix(prim)
        # trans = pose.ExtractTranslation()
        # quat = pose.ExtractRotation().GetQuaternion()
        # print(trans, quat)

        if world.current_time_step_index == 0:
            world.reset()

#     simulation_context.step(render=True)

# simulation_context.stop()
simulation_app.close()
