# -*- coding: utf-8 -*-

from utils import *
#import roi_ae_lstm_v6 as roi
import ae_lstm_v2 as tr
import SIM_ROI as S
import numpy as np
import cv2
import matplotlib.pyplot as plt

trainer = tr.AE_LSTM_Trainer()
trainer.prepare_for_test(load_val_data=True)

capture_size = (180, 320)
#view_params = ([0,-0.9,1.35], [0,-0.55,0.5], [0,0,1])
view_params = [0,-0.7,1.4], [0,-0.6,0.5], [0,0,1]

env = S.SIM_ROI(scene='reaching', is_vr=False)
control = S.VRController_ROI(0)
cam = S.CAMERA_ROI(capture_size[1], capture_size[0], fov=50)
cam.setViewMatrix(*view_params)

#S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(False)

def normalize_joint_position(q):
    return trainer.val_ds.normalize_joint_position(q)

def unnormalize_joint_position(q):
    return trainer.val_ds.unnormalize_joint_position(q)

def toCart(jv, unnormalize=True):
    jv = unnormalize_joint_position(jv)
    env.moveArm(jv[:6])
    sync()
    return getEFPos()

def toCartTraj(jvs):
    return [toCart(jv) for jv in jvs]

def sync(steps=100):
    for j in range(steps):
        S.p.stepSimulation()


sm = StateManager(tr.time_window_size)

def reset():
    env.setInitialPos()
    sync()
    img0 = captureRGB()
    js0 = env.getJointState()
    js0 = normalize_joint_position(js0)
    sm.initializeHistory(img0, js0)

def captureRGB():
    img = cam.getImg()[2][:,:,:3]
    img = cv2.resize(img, swap(tr.input_image_shape[:2]))
    return img / 255.

def captureSegImg():
    img = cam.getImg()[4]
    img = cv2.resize(img, swap(tr.input_image_shape[:2]))
    return img / 255.

def getEFPos():
    s = S.p.getLinkState(env.ur5, 7)
    return s[0]

def run(max_steps=50, anim_gif=False):
    for i in range(max_steps):
        print('i = ', i)
        img = captureRGB()
        js = env.getJointState()

        js = normalize_joint_position(js)
        sm.addState(img, js)
        res = trainer.model.predict(sm.getHistory())
        if len(res) == 3:
            imgs, jvs, rois = res
        else:
            imgs, jvs = res
        jv = jvs[0]
        jv = unnormalize_joint_position(jv)

        print(jv)
        env.moveArm(jv[:6])
        sync()

    if anim_gif:
        create_anim_gif_from_images(sm.getFrameImages(), 'run.gif')

def predict_sequence_open(group_num=0):
    trajs = trainer.predict_sequence_open(group_num)
    return map(lambda x: np.array(toCartTraj(x)), trajs)

def visualize_predicted_vectors(groups=range(10)):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    coordinate(ax, [-0.6,0.6],[-1.0,-0.2])
    for group in groups:
        cs,ps,ls = predict_sequence_open(group)
        draw_predictions_and_labels(ax, cs, ps, ls)
    plt.show()

class CamState:
    def __init__(self, fov=50, view_params=view_params):
        self.fov = fov
        self.view_params = view_params

class SimState:
    def __init__(self, object_pose, target_pose, robot_joint_positions):
        self.objectPose = object_pose
        self.targetPose = target_pose
        self.robotJointPositions = robot_joint_positions
        # self.cam = S.CAMERA_ROI(width, height, fov=50)

def saveState(env):
    box_pose = S.p.getBasePositionAndOrientation(env.box)
    target_pose = S.p.getBasePositionAndOrientation(env.target)
    js = env.getJointState()
    return SimState(box_pose, target_pose, js)

def loadState(env, sim_state):
    js = sim_state.robotJointPositions
    env.moveArm(js[:6])
    env.moveGripper(js[6])
    sync()
    S.p.resetBasePositionAndOrientation(env.box, sim_state.objectPose[0], sim_state.objectPose[1])
    S.p.resetBasePositionAndOrientation(env.target, sim_state.targetPose[0], sim_state.targetPose[1])

s = SimState(
    ((-0.0702149141398469, -0.6386324902206956, 0.7774899999999993),
         (-2.78783798812622e-17, -2.4409568836878583e-16, 0.00025199415600674166, 0.9999999682494721)),
    ((0.27426512264674074, -0.5566910719253673, 0.77), (0.0, 0.0, 0.0, 1.0)),
    [-0.65681903, -0.63648103,  1.38169039, -1.01598524, -0.50431903, 0.23976613,
         0, 0, 0, 0])

