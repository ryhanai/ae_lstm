# -*- coding: utf-8 -*-

import math
import time
import os
import copy
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.utils import *
from sim.pybullet_tools import *
from sim import SIM_KITTING as S
from sim import domain_randomization


# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-s', '--scene', type=str, default='basket_filling_scene.yaml')
# parser.add_argument('-b', '--baseline', action='store_true')
# parser.add_argument('-u', '--ui', type=str, default='none')
# args = parser.parse_args()

# message('scene = {}'.format(args.scene))
# message('ui = {}'.format(args.ui))

scene = 'basket_filling_scene.yaml'

use_rviz = False

if use_rviz:
    import force_distribution_viewer
    viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

env = S.SIM(scene_file=scene, rootdir='../../')
# S.p.changeDynamics(env.robot, 23, collisionMargin=0.0)
S.p.changeDynamics(env.target, 0, collisionMargin=-0.03)
S.p.changeDynamics(env.target, 1, collisionMargin=-0.03)
# S.p.setCollisionFilterPair(env.robot, env.cabinet, 23, 0, 0)
S.p.changeVisualShape(env.target, -1, specularColor=[0, 0, 0])

S.p.setRealTimeSimulation(True)

cam = env.getCamera('camera1')
fcam = env.getCamera('force_camera1')
rec = S.RECORDER_KITTING(cam.getCameraConfig())

objects = {
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
}

center_of_objects = {
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


preferred_poses = {
    # (pose, center of placing area)
    '004_sugar_box': [([0, 0, 0], [0.01, 0.015]), ([math.pi / 2, -math.pi / 2, 0], [-0.02, 0.08]), ([-math.pi/2, -math.pi/2, 0], [0.02, -0.008])],
    '005_tomato_soup_can': [([0, 0, 0], [0.01, -0.09]), ([0, -math.pi/2, 0], [0.05, -0.09])],
    '007_tuna_fish_can': [([0, 0, 0], [0.02, 0.02]), ([0, -math.pi/2, 0], [0.02, 0.02])],
    '008_pudding_box': [([0, 0, -0.5], [-0.01, -0.02]), ([-math.pi/2, -0.5, 0], [-0.01, -0.02])],
    '009_gelatin_box': [([0, 0, -0.25], [0.02, 0]), ([math.pi/2, 0.2, 0], [0.02, 0.015])],
    '010_potted_meat_can': [([0, 0, 0], [0.035, 0.025]), ([math.pi/2, 0, 0], [0.035, 0.035])]
}

valid_object_ids = []
object_cache = {}
object_id_table = {-1: 0, 0: 0, 1: 1, 2: 2}  # -1:???, 0:???, 1:table, 2:bin


class ObjectProxy:
    def __init__(self, name, mass, default_pose):
        self.name = name
        self.mass = mass
        self.default_pose = default_pose
        self.load_mesh(name, mass)
        self.reset_pose()
        S.p.changeDynamics(self.body, -1, lateralFriction=0.3, rollingFriction=0.01, activationState=S.p.ACTIVATION_STATE_DISABLE_WAKEUP)
        self.used = False

    def load_mesh(self, name, mass):
        mesh_path = 'specification/meshes/objects/ycb/{}/google_16k/textured.obj'.format(name)
        self.body = create_mesh_body(mesh_path, mass=mass)

    def get_body(self):
        return self.body

    def unload(self):
        S.p.removeBody(self.body)

    def activate(self):
        self.used = True

    def deactivate(self):
        self.used = False

    def isActive(self):
        return self.used

    def reset_pose(self):
        set_pose(self.body, self.default_pose)


def get_object_center(name):
    body = object_cache.get(name).get_body()
    pose = S.get_pose(body)
    center = center_of_objects[name]
    return env.multiplyTransforms((center, [0,0,0,1]), pose)[0]


def load_objects():
    x = 0
    for name, mass in objects.items():
        pose = np.array([x, 0, 0]), unit_quat()
        object_cache[name] = ObjectProxy(name, mass, pose)
        x += 0.1
    for k, oc in object_cache.items():
        oid = int(k.split('_')[0])
        object_id_table[oc.get_body()] = oid
    cam = env.getCamera('camera1')
    cam.setSegmentationIdMap(object_id_table)


def sample_place_pose():
    xy = np.array([0.17, 0.10]) * (np.array([-0.5, -0.5]) + np.random.random(2))
    z = 0.73 + 0.25
    return (np.append(xy, z), unit_quat())


def sample_place_pose2(name):
    smpl_params = preferred_poses.get(name)
    if smpl_params is None:
        xy = np.array([0.16, 0.09]) * (np.array([-0.5, -0.5]) + np.random.random(2))
        q = unit_quat()
    else:
        smpl_param = smpl_params[np.random.randint(len(smpl_params))]
        xy = np.array([0.16, 0.09]) * (np.array([-0.5, -0.5]) + np.random.random(2)) + np.array(smpl_param[1])
        q = S.p.getQuaternionFromEuler(smpl_param[0])
    z = 0.73 + 0.25
    return np.append(xy, z), q


def place_by_drop(name, pose):
    global valid_object_ids
    body = create_mesh_body('specification/meshes/objects/ycb/{}/google_16k/textured.obj'.format(name), mass=objects[name])
    set_pose(body, pose)
    # valid_object_ids.append((name, body))
    t0 = time.time()
    while time.time() - t0 < 5.0:
        observe(n_frames=1)
        scene_is_stable = True
        for n,id in valid_object_ids:
            v, w = S.p.getBaseVelocity(id)
            v = np.linalg.norm(v)
            w = np.linalg.norm(w)
            print('{}\n vel = {}, avel = {}'.format(n, v, w))
            if v > 0.01 or w > 1.0:
                scene_is_stable = False
            if scene_is_stable:
                break
        observe(n_frames=10)
    return body


def place(name, pose):
    global valid_object_ids
    S.p.setRealTimeSimulation(False)
    oc = object_cache.get(name)  
    body = oc.get_body()
    set_pose(body, pose)
    oc.activate()

    while True:
        p,q = pose
        p[2] -= 0.005
        pose = p,q
        set_pose(body, pose)
        S.p.performCollisionDetection()
        cps = S.p.getContactPoints(body)
        if len(cps) > 0:
            break

    S.p.setRealTimeSimulation(True)
    t0 = time.time()
    while time.time() - t0 < 5.0:
        observe(n_frames=1)
        scene_is_stable = True
        for n,id in valid_object_ids:
            v,w = S.p.getBaseVelocity(id)
            v = np.linalg.norm(v)
            w = np.linalg.norm(w)
            print('{}\n vel = {}, avel = {}'.format(n, v, w))
            if v > 0.01 or w > 1.0:
                scene_is_stable = False
            if scene_is_stable:
                break
    observe(n_frames=10)
    return body


def place_object(name):
    place(name, sample_place_pose2(name))


def create_random_scene(n_objects=3, scene_writer=None, randomization=False):
    selected_objects = np.random.choice(list(objects.keys()), n_objects, replace=False)  # sample with no duplication
    print(selected_objects)
    for object in selected_objects:
        place_object(object)
        if randomization:
            randomize_scene()
        if scene_writer is not None:
            scene_writer.save_scene()


def clear_scene():
    for k, v in object_cache.items():
        if v.isActive():
            v.deactivate()
            v.reset_pose()


def get_bin_state():
    bin_state = []
    for k, v in object_cache.items():
        if v.isActive():
            body = v.get_body()
            bin_state.append((k, get_pose(body)))
    return bin_state


def observe(n_frames=10, moving_average=True):
    for i in range(n_frames):
        cam.getImg()
        f = fcam.getDensity(moving_average=moving_average, reshape_result=True)[1]
        if use_rviz:
            viewer.publish_bin_state(get_bin_state(), fcam.positions, f, draw_fmap=True, draw_force_gradient=False)


class SceneWriter:
    def __init__(self):
        self.groupNo = self.scanDataDirectory()
        self.frameNo = 0

    def scanDataDirectory(self):
        return 0

    def createNewGroup(self):
        self.groupNo += 1
        self.frameNo = 0
        self.group_dir = 'data/{:d}'.format(self.groupNo)
        if not os.path.exists(self.group_dir):
            os.makedirs(self.group_dir)

    def save_scene(self):
        imgs = cam.getImg()
        rgb = imgs[2]
        depth = imgs[3]
        seg = imgs[4]
        force = fcam.getDensity(moving_average=True, reshape_result=True)[1]
        cv2.imwrite(os.path.join(self.group_dir, 'rgb{:05d}.jpg'.format(self.frameNo)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pd.to_pickle(depth, os.path.join(self.group_dir, 'depth_zip{:05d}.pkl'.format(self.frameNo)), compression='zip')
        cv2.imwrite(os.path.join(self.group_dir, 'seg{:05d}.png'.format(self.frameNo)), seg)
        pd.to_pickle(force, os.path.join(self.group_dir, 'force_zip{:05d}.pkl'.format(self.frameNo)), compression='zip')
        pd.to_pickle(get_bin_state(), os.path.join(self.group_dir, 'bin_state{:05d}.pkl'.format(self.frameNo)))
        self.frameNo += 1


def create_dataset(n_sequence=1000, n_objects_in_a_scene=6, randomization=True):
    sw = SceneWriter()
    for n in range(n_sequence):
        sw.createNewGroup()
        create_random_scene(n_objects_in_a_scene, scene_writer=sw, randomization=randomization)
        clear_scene()


def apply_randomization(config, camera):
    S.p.configureDebugVisualizer(lightPosition=config['light_position'])
    S.p.configureDebugVisualizer(shadowMapIntensity=config['shadow_map_intensity'])
    S.p.configureDebugVisualizer(shadowMapResolution=config['shadow_map_resolution'])
    S.p.configureDebugVisualizer(shadowMapWorldSize=config['shadow_map_world_size'])
    image_size, fov, aspect_ratio = config['camera_intrinsics']
    camera.setProjectionMatrix(width=image_size[0], height=image_size[1], fov=fov, near=0.1, far=2.0, aspect=aspect_ratio)
    camera.setViewMatrix(*config['camera_extrinsics'])

    for o in config['objects']:
        name = o['name']
        body_id = object_cache[name].body
        S.p.changeVisualShape(body_id, -1, rgbaColor=o['color'])
        S.p.changeVisualShape(body_id, -1, specularColor=o['specular_color'])


def randomize_scene():
    cam = env.getCamera('camera1')
    camera_conf0 = copy.copy(cam.cameraConfig)
    config = domain_randomization.sample_scene_parameters(object_cache, cam.cameraConfig)
    apply_randomization(config, cam)
    cam.cameraConfig = camera_conf0  # restore camera configuration


def randomization_test(n=10):
    for i in range(n):
        randomize_scene()
        cam = env.getCamera('camera1')
        img = cam.getImg()[2]
        plt.imshow(img)
        plt.savefig('{:05d}.png'.format(i))
