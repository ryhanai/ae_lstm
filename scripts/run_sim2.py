# -*- coding: utf-8 -*-

import numpy as np
import cv2, argparse
import matplotlib.pyplot as plt

from core.utils import *
import SIM_ROI as S


parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--scene', type=str, default='reaching_scene.yaml')
parser.add_argument('-b', '--baseline', action='store_true')
args = parser.parse_args()

if args.baseline:
    message('loading baseline model')
    import ae_lstm_v3 as mdl # baseline w/o ROI, raeching
else:
    message('loading ROI model')
    import roi_ae_lstm_v15 as mdl
    #import roi_ae_lstm_v16 as mdl

message('scene = {}'.format(args.scene))
    
tr = mdl.prepare_for_test()

env = S.SIM_ROI(scene_file=args.scene, is_vr=False)
control = S.VRController_ROI(0)
cam = env.getCamera('camera1')

#S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(False)

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


sm = StateManager(mdl.time_window_size)
roi_hist = []
predicted_images = []

def reset(target_pose=None):
    global roi_hist
    global predicted_images

    env.resetRobot()
    if target_pose == None:
        target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
        target_ori = S.p.getQuaternionFromEuler([0,0,0])
    else:
        target_pos, target_ori = target_pose
    env.setObjectPosition(target_pos, target_ori)
    sync()

    img0 = captureRGB()
    js0 = env.getJointState()
    js0 = normalize_joint_position(js0)
    sm.initializeHistory(img0, js0)
    roi_hist = []
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

class CamState:
    def __init__(self, fov=50):
        self.fov = fov
        c = cam.getCameraConfig()
        self.view_params = [c['eyePosition'], c['targetPosition'], c['upVector']]

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

