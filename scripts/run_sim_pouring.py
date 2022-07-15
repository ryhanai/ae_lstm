# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2, argparse
import matplotlib.pyplot as plt

from core.utils import *
import SIM_POURING as S


parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--scene', type=str, default='reaching_scene.yaml')
parser.add_argument('-b', '--baseline', action='store_true')
args = parser.parse_args()


message('scene = {}'.format(args.scene))
    

env = S.SIM_ROI(scene_file=args.scene, is_vr=False)
control = S.VRController_ROI(False, use3Dmouse=False)
cam = env.getCamera('camera1')

#S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(True)

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

def sync(steps=100):
    for j in range(steps):
        S.p.stepSimulation()


# sm = StateManager(mdl.time_window_size)
roi_hist = []
predicted_images = []

def reset(target_pose=None):
    global roi_hist
    global predicted_images

    env.resetRobot()
    if target_pose == None:
        # 'reaching_scene'
        # target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
        # target_ori = S.p.getQuaternionFromEuler([0,0,0])

        # 'reaching_2ways_scene'
        while True:
            target_pos = np.append([-0.4, -0.75] + [0.75, 0.25] * np.random.random(2), 0.79)
            target_ori = S.p.getQuaternionFromEuler([0,0,0])
            if not (-0.15 < target_pos[0] and target_pos[0] < 0.15 and target_pos[1] > -0.65):
                break
    else:
        target_pos, target_ori = target_pose
    env.setObjectPosition(target_pos, target_ori)
    sync()

    img0 = captureRGB()
    js0 = env.getJointState()
    js0 = normalize_joint_position(js0)
    sm.initializeHistory(img0, js0)
    roi_hist = []
    env.clearFrames()
    predicted_images = []

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

def calc_train_data_bias(groups=range(1,5)):
    results = []
    for g in groups:
        s = pd.read_pickle('~/Dataset/dataset2/reaching/{}/sim_states.pkl'.format(g))[1][-1]
        q = s['jointPosition']
        p_target = np.array(s['target'][0])
        env.setObjectPosition(p_target)
        env.moveArm(q[:6])
        sync()
        p_tcp = env.getTCPXYZ()
        results.append((p_target, p_tcp))
    return results

def rerender(groups=range(1,2), task='reaching'):
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
        
n_runs = 0
        
def run(max_steps=80, anim_gif=False):
    global n_runs
    
    for i in range(max_steps):
        print('i = ', i)
        img = captureRGB()
        js = env.getJointState()

        js = normalize_joint_position(js)
        sm.addState(img, js)
        res = tr.predict(sm.getHistory())
        if len(res) == 3:
            imgs, jvs, rois = res
            if rois.ndim == 3:
                roi_hist.append(rois[0][0])
            else:
                roi_hist.append(rois[0])
        else:
            imgs, jvs = res

        predicted_images.append(imgs[0])
        # plt.imshow(imgs[0])
        # plt.pause(.01)

        jv = jvs[0]
        jv = unnormalize_joint_position(jv)

        print(jv)
        env.moveArm(jv[:6])
        sync()

    if anim_gif:
        create_anim_gif_from_images(sm.getFrameImages(), 'run{:0=5}.gif'.format(n_runs), roi_hist, predicted_images)
    n_runs += 1


def run_test():
    for i, pose in enumerate(env.getSceneDescription()['test']):
        message(i, tag='[TEST]: ')
        reset((pose['xyz'], S.p.getQuaternionFromEuler(pose['rpy'])))
        run(80, anim_gif=True)

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


def task_completed():
    p_tcp = env.getTCPXYZ()
    p_target = np.array(S.p.getBasePositionAndOrientation(env.target)[0])
    return scipy.linalg.norm(p_tcp[:2]-p_target[:2], 2) < 0.02
    
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

                
                
