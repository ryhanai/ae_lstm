# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2, argparse
import matplotlib.pyplot as plt

from core.utils import *
import SIM_KITTING as S
import tf
from pybullet_tools import *

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

objects = [
  '004_sugar_box',
  '006_mustard_bottle',
  '007_tuna_fish_can',
  '010_potted_meat_can',
  '026_sponge', # soft body
  '011_banana',
  '012_strawberry',
  '013_apple',
  '016_pear',
]

def sample_place_pose():
  xy = 0.1 * (np.array([-0.5,-0.5]) + np.random.random(2))
  z = 0.785+0.3
  return (np.append(xy, z), unit_quat())

valid_object_ids = []

def place(name, pose):
  global valid_object_ids
  body = create_mesh_body('specification/meshes/objects/ycb/{}/google_16k/textured.obj'.format(name))
  set_pose(body, pose)
  valid_object_ids.append((name, body))
  t0 = time.time()
  while time.time() - t0 < 2.0:
    observe(n_frames=10)
  return body

def place_object(object):
  place(object, sample_place_pose())

def create_random_scene(n_objects=3):
  selected_objects = np.random.choice(objects, n_objects)
  print(selected_objects)
  for object in selected_objects:
    place_object(object)

def clear_scene():
  global valid_object_ids
  for n,i in valid_object_ids:
    S.p.removeBody(i)
    valid_object_ids = []

from publish_force_distribution import *
    
def observe(n_frames=10, moving_average=True):
  for i in range(n_frames):
    cam.getImg()
    # fcam.getImg()
    positions, fd = fcam.getDensity(moving_average)
    bin_state = []
    for name, oid in valid_object_ids:
      bin_state.append((name, get_pose(oid)))
    publish_bin_state(bin_state, (positions, fd))
    rate.sleep
    
