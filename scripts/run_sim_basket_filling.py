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
  '006_mustard_bottle': 0.603,
  '007_tuna_fish_can': 0.171,
  '010_potted_meat_can': 0.370,
  '026_sponge': 0.0062,
  '011_banana': 0.066,
  '012_strawberry': 0.018,
  '013_apple': 0.068,
  '016_pear': 0.049,
}

def sample_place_pose():
  xy = 0.1 * (np.array([-0.5,-0.5]) + np.random.random(2))
  z = 0.785+0.3
  return (np.append(xy, z), unit_quat())

valid_object_ids = []

def place(name, pose):
  global valid_object_ids
  body = create_mesh_body('specification/meshes/objects/ycb/{}/google_16k/textured.obj'.format(name), mass=objects[name])
  set_pose(body, pose)
  valid_object_ids.append((name, body))
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

def place_object(object):
  place(object, sample_place_pose())

def create_random_scene(n_objects=3):
  selected_objects = np.random.choice(list(objects.keys()), n_objects)
  print(selected_objects)
  for object in selected_objects:
    place_object(object)

def clear_scene():
  global valid_object_ids
  for n,i in valid_object_ids:
    S.p.removeBody(i)
    valid_object_ids = []

from publish_force_distribution import *

def get_bin_state():
  bin_state = []
  for name, oid in valid_object_ids:
    bin_state.append((name, get_pose(oid)))
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
    return 557

  def createNewGroup(self):
    self.groupNo += 1
    self.frameNo = 0
    self.group_dir = 'data/{:d}'.format(self.groupNo)
    if not os.path.exists(self.group_dir):
      os.makedirs(self.group_dir)

  def save_scene(self):
    rgb = cam.getImg()[2]
    force = fcam.getDensity(moving_average=True, reshape_result=True)[1]
    plt.imsave(os.path.join(self.group_dir, 'rgb{:05d}.jpg'.format(self.frameNo)), rgb)
    pd.to_pickle(force, os.path.join(self.group_dir, 'force{:05d}.pkl'.format(self.frameNo)))
    pd.to_pickle(get_bin_state(), os.path.join(self.group_dir, 'bin_state{:05d}.pkl'.format(self.frameNo)))
    self.frameNo += 1

def visualize_forcemaps(force_distribution):
    f = force_distribution / np.max(force_distribution)
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.1)

    channels = f.shape[-1]
    for p in range(channels):
        ax = fig.add_subplot(channels//10, 10, p+1)
        ax.axis('off')
        ax.imshow(f[:,:,p], cmap='gray', vmin=0, vmax=1.0)
    plt.show()

def create_dataset(n_sequence=3):
  sw = SceneWriter()
  for n in range(n_sequence):
    sw.createNewGroup()
    for i in range(4):
      create_random_scene(1)
      sw.save_scene()
    clear_scene()
