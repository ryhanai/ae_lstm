# -*- coding: utf-8 -*-

import time, math
import numpy as np
import scipy.linalg
import cv2
import matplotlib.pyplot as plt

from core.utils import *
import SIM_PEG as S
#import roi_ae_lstm_v14 as mdl
import ae_lstm_v3 as mdl

tr = mdl.prepare_for_test()

capture_size = (180, 320)
view_params = [0,-1.15,1.55], [0,-0.42,0.5], [0,0,1]

env = S.SIM_ROI(scene='reaching', is_vr=False)
control = S.VRController_ROI(False, use3Dmouse=False)
cam = S.CAMERA_ROI(capture_size[1], capture_size[0], fov=40, shadow=False)
cam.setViewMatrix(*view_params)

S.p.setRealTimeSimulation(False)

def normalize_joint_position(q):
    return tr.val_ds.normalize_joint_position(q)

def unnormalize_joint_position(q):
    return tr.val_ds.unnormalize_joint_position(q)

def toCart(jv, unnormalize=True):
    jv = unnormalize_joint_position(jv)
    env.moveArm(jv[:6])
    sync()
    return getEFPos()

def toCartTraj(jvs):
    return [toCart(jv) for jv in jvs]

def sync(steps=100, wait=False):
    for j in range(steps):
        S.p.stepSimulation()
        if wait:
            time.sleep(wait)


sm = StateManager(mdl.time_window_size)
roi_hist = []

def reset():
    global roi_hist
    env.setInitialPos()
    #sync()
    img0 = captureRGB()
    js0 = env.getJointState()
    js0 = normalize_joint_position(js0)
    sm.initializeHistory(img0, js0)
    roi_hist = []
    env.clearFrames()

def captureRGB():
    img = cam.getImg()[2][:,:,:3]
    img = cv2.resize(img, swap(mdl.input_image_size))
    return img / 255.

def captureSegImg():
    img = cam.getImg()[4]
    img = cv2.resize(img, swap(mdl.input_image_size))
    return img / 255.

def getEFPos():
    s = S.p.getLinkState(env.ur5, 7)
    return s[0]

def generate_eval_goal_acc_samples(n_samples):
    samples = []
    np.random.seed(0)
    for i in range(n_samples):
        target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.77)
        samples.append(target_pos)
    return samples

import scipy.linalg

def eval_goal_accuracy(n_samples=20):
    reset()
    samples = generate_eval_goal_acc_samples(n_samples)
    results = []
    for p in samples:
        env.resetRobot()
        env.setObjectPosition(p)
        sync()
        run()
        p1 = env.getTargetXYZ()
        p2 = env.getTCPXYZ()
        #d = scipy.linalg.norm(p1-p2, ord=2)
        results.append((p1,p2))
    return results

import pandas as pd

def rerender(groups=range(1,2), task='peg-in-hole'):
    for g in groups:
        env.clearFrames()
        env.resetRobot()
        for s in pd.read_pickle('~/Dataset/dataset2/{}/{}/sim_states.pkl'.format(task, g))[1]:
            q = s['jointPosition']
            p_target = np.array(s['target'][0])
            env.setObjectPosition(p_target)
            env.moveArm(q[:6])
            sync()
            js = env.getJointState()
            img = cam.getImg()
            print('save:[{}]: {}'.format(env.frameNo, js))
            d = {'frameNo':env.frameNo, 'jointPosition':js, 'image':img}
            for k,id in env.objects.items():
                d[k] = S.p.getBasePositionAndOrientation(id)
            env.frames.append(d)
            env.frameNo += 1
        env.groupNo = g
        env.writeFrames(cam.getCameraConfig())

def run(max_steps=50, anim_gif=False, anim_gif_file='run.gif'):
    for i in range(max_steps):
        print('i = ', i)
        img = captureRGB()
        js = env.getJointState()

        js = normalize_joint_position(js)
        sm.addState(img, js)
        res = tr.model.predict(sm.getHistory())
        if len(res) == 3:
            imgs, jvs, rois = res
            roi_hist.append(rois[0][0])
        else:
            imgs, jvs = res
        jv = jvs[0]
        jv = unnormalize_joint_position(jv)

        print(jv)
        env.moveArm(jv[:6])
        sync()

    if anim_gif:
        create_anim_gif_from_images(sm.getFrameImages(), anim_gif_file, roi_hist)

def predict_sequence_open(group_num=0):
    trajs = tr.predict_sequence_open(group_num)
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

def task_completed():
    p_tcp = env.getTCPXYZ()
    p_target = np.array(S.p.getBasePositionAndOrientation(env.target)[0])
    return p_tcp[2] < 0.876 and scipy.linalg.norm(p_tcp[:2]-p_target[:2], 2) < 0.06
    
def teach():
    #S.p.setTimeStep(1./240)
    S.p.setRealTimeSimulation(True)

    while True:
        img = cam.getImg()
        control.update()
        env.saveFrame(img)

        # finish task?
        if task_completed():
            env.writeFrames(cam.getCameraConfig())
            reset()
            continue

        # if control.B1:  #VR
        #     gripperOri = [1.75,0,1.57]
        #     gripperOriQ = calc.get_QfE(gripperOri)
        #     joy_pos = control.position
        #     val = calc.get_IK(env.ur5, 8,joy_pos, gripperOriQ)
        #     invJoints = [1,2,3,4,5,6,13,15,17,18,20,22]
        #     env.setJointValues(env.ur5,invJoints,val)
        if control.BLK:
            break
        if control.B2:
            reset()
            continue

        v = 0.008
        vx = vy = math.sqrt(v*v / 2.)
        vtheta = 3

        if control.use3Dmouse:
            env.moveEF([v * (-control.last_msg.y), v * control.last_msg.x, v * control.last_msg.z])
        else:
            if control.KTL:
                env.moveEF([-v, 0, 0])
            if control.KTR:
                env.moveEF([v, 0, 0])
            if control.KTU:
                env.moveEF([0, v, 0])
            if control.KTD:
                env.moveEF([0, -v, 0])
            if control.KTLU:
                env.moveEF([-vx, vy, 0])
            if control.KTLD:
                env.moveEF([-vx, -vy, 0])
            if control.KTRU:
                env.moveEF([vx, vy, 0])
            if control.KTRD:
                env.moveEF([vx, -vy, 0])
            if control.KD:
                env.moveEF([0, 0, -v])
            if control.KU:
                env.moveEF([0, 0, v])
            if control.CG:
                env.closeGripper()
            if control.OG:
                env.openGripper()
