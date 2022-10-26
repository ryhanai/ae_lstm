# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2, argparse
import matplotlib.pyplot as plt

from core.utils import *
import SIM_KITTING as S
import tf
from pybullet_tools import *
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--scene', type=str, default='basket_filling_scene.yaml')
parser.add_argument('-b', '--baseline', action='store_true')
parser.add_argument('-u', '--ui', type=str, default='none')
args = parser.parse_args()

message('scene = {}'.format(args.scene))
message('ui = {}'.format(args.ui))

env = S.SIM(scene_file=args.scene)
#S.p.changeDynamics(env.robot, 23, collisionMargin=0.0)
S.p.changeDynamics(env.target, 0, collisionMargin=-0.03)
S.p.changeDynamics(env.target, 1, collisionMargin=-0.03)
#S.p.setCollisionFilterPair(env.robot, env.cabinet, 23, 0, 0)

S.p.setRealTimeSimulation(True)

cam = env.getCamera('camera1')
fcam = env.getCamera('force_camera1')
rec = S.RECORDER_KITTING(cam.getCameraConfig())

objects = {
  '004_sugar_box': 0.514,
  '005_tomato_soup_can': 0.349,
  #'006_mustard_bottle': 0.603,
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
  #'023_wine_glass': 0.133, # no 3D model
  '026_sponge': 0.0062,
  '040_large_marker': 0.0158,
  #'041_small_marker': 0.0082,
  '055_baseball': 0.148,
  '056_tennis_ball': 0.056,
  '061_foam_brick': 0.059,
}

preferred_poses = {
  # (pose, center of placing area)
  '004_sugar_box': [([0,0,0],[0.01,0.015]), ([0,-math.pi/2,0],[0.09,0.015])],
  '005_tomato_soup_can': [([0,0,0],[0.01,-0.09]), ([0,-math.pi/2,0],[0.05,-0.09])],
  '007_tuna_fish_can': [([0,0,0],[0.02,0.02]), ([0,-math.pi/2,0],[0.02,0.02])],
  '008_pudding_box': [([0,0,-0.5],[-0.01,-0.02]), ([-math.pi/2,-0.5,0],[-0.01,-0.02])],
  '009_gelatin_box': [([0,0,-0.25],[0.02,0]), ([math.pi/2,0.2,0],[0.02,0.015])],
  '010_potted_meat_can': [([0,0,0],[0.035,0.025]), ([math.pi/2,0,0],[0.035,0.035])]
}

valid_object_ids = []
object_cache = {}
object_id_table = {0:0, 1:1, 2:2} # 0:???, 1:table, 2:bin

class ObjectProxy:
  def __init__(self, name, mass, default_pose):
    self.name = name
    self.mass = mass
    self.default_pose = default_pose
    self.load_mesh(name, mass)
    self.reset_pose()
    S.p.changeDynamics(self.body, -1, lateralFriction=0.3, activationState=S.p.ACTIVATION_STATE_DISABLE_WAKEUP)
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


def load_objects():
  x = 0
  for name,mass in objects.items():
    pose = np.array([x,0,0]), unit_quat()
    object_cache[name] = ObjectProxy(name, mass, pose)
    x += 0.1
  for k,oc in object_cache.items():
    oid = int(k.split('_')[0])
    object_id_table[oc.get_body()] = oid
  cam = env.getCamera('camera1')
  cam.setSegmentationIdMap(object_id_table)

def sample_place_pose():
  xy = np.array([0.17, 0.10]) * (np.array([-0.5,-0.5]) + np.random.random(2))
  z = 0.73+0.25
  return (np.append(xy, z), unit_quat())

def sample_place_pose2(name):
  smpl_params = preferred_poses.get(name)
  if smpl_params == None:
    xy = np.array([0.17, 0.10]) * (np.array([-0.5,-0.5]) + np.random.random(2))
    q = unit_quat()
  else:
    smpl_param = smpl_params[np.random.randint(len(smpl_params))]
    xy = np.array([0.17, 0.10]) * (np.array([-0.5,-0.5]) + np.random.random(2)) + np.array(smpl_param[1])
    q = S.p.getQuaternionFromEuler(smpl_param[0])
  z = 0.73+0.25
  return (np.append(xy, z), q)

def place_by_drop(name, pose):
  global valid_object_ids
  body = create_mesh_body('specification/meshes/objects/ycb/{}/google_16k/textured.obj'.format(name), mass=objects[name])
  set_pose(body, pose)
  #valid_object_ids.append((name, body))
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

def create_random_scene(n_objects=3, scene_writer=None):
  selected_objects = np.random.choice(list(objects.keys()), n_objects, replace=False) # sample with no duplication
  print(selected_objects)
  for object in selected_objects:
      place_object(object)
      if scene_writer != None:
        scene_writer.save_scene()

def clear_scene():
  for k,v in object_cache.items():
    if v.isActive():
      v.deactivate()
      v.reset_pose()
      
from publish_force_distribution import *

def get_bin_state():
  bin_state = []
  for k,v in object_cache.items():
    if v.isActive():
      body = v.get_body()
      bin_state.append((k, get_pose(body)))
  return bin_state

def observe(n_frames=10, moving_average=True):
  for i in range(n_frames):
    cam.getImg()
    publish_bin_state(get_bin_state(), fcam.getDensity(moving_average))
    rate.sleep

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
    plt.imsave(os.path.join(self.group_dir, 'rgb{:05d}.jpg'.format(self.frameNo)), rgb)
    plt.imsave(os.path.join(self.group_dir, 'depth{:05d}.png'.format(self.frameNo)), depth)
    plt.imsave(os.path.join(self.group_dir, 'seg{:05d}.png'.format(self.frameNo)), seg)
    pd.to_pickle(force, os.path.join(self.group_dir, 'force_zip{:05d}.pkl'.format(self.frameNo)), compression='zip')
    pd.to_pickle(get_bin_state(), os.path.join(self.group_dir, 'bin_state{:05d}.pkl'.format(self.frameNo)))
    self.frameNo += 1

def create_dataset(n_sequence=1000, n_objects_in_a_scene=6):
  sw = SceneWriter()
  for n in range(n_sequence):
    sw.createNewGroup()
    create_random_scene(n_objects_in_a_scene, scene_writer=sw)
    clear_scene()
