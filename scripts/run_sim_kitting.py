# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2, argparse
import matplotlib.pyplot as plt

from core.utils import *
import SIM_KITTING as S
import tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--scene', type=str, default='kitting_scene.yaml')
parser.add_argument('-b', '--baseline', action='store_true')
parser.add_argument('-u', '--ui', type=str, default='none')
args = parser.parse_args()

message('scene = {}'.format(args.scene))
message('ui = {}'.format(args.ui))

#tr = mdl.prepare_for_test()

env = S.SIM(scene_file=args.scene)
S.p.changeDynamics(env.robot, 23, collisionMargin=0.0)
S.p.changeDynamics(env.target, 0, collisionMargin=-0.01)
S.p.changeDynamics(env.target, 1, collisionMargin=-0.01)

#S.p.setCollisionFilterPair(env.robot, env.cabinet, 23, 0, 0)
cam = env.getCamera('camera1')
fcam = env.getCamera('force_camera1')
rec = S.RECORDER(cam.getCameraConfig())

if args.ui == '3dmouse':
    import SpaceNavUI
    ui = SpaceNavUI.SpaceNavUI()

def eventLoop():
    while True:
        e = S.p.getMouseEvents()

def teach():
    #S.p.setTimeStep(1./240)
    S.p.setRealTimeSimulation(True)

    while True:
        img = cam.getImg()
        fimg = fcam.getImg()
        #control.update()
        js = env.getJointState()
        rec.saveFrame(img, fimg, js, env)

        if ui.getEventSignal() == SpaceNavUI.UI_EV_RESET:
            rec.writeFrames()
            reset()
            continue

        v,w = ui.getControlSignal()
        w = S.p.getQuaternionFromEuler(w)
        env.moveEF(v, w)

def teach_procedurally():
    pass

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# Goal State
# pen (robot, 23)
tf_approach = ((0.03664122521408607, -0.5315819169452275, 0.963800216862066),
               (0.25350660559819876,
                0.46699976144543603,
                -0.5618843155322776,
                0.6339807881054513))

# target 
tf_target = ((0.03975343991738431, -0.6731654428797486, 0.79),
             (0.0, 0.0, -0.6185583387305793, 0.785738876209435))


# approach waypoint relative to tf_target
#tf_target_approach = ([-0.13951663,  0.01909836,  0.17542246],
#                    [-0.02402769,  0.52414153, -0.01857238,  0.85108954])
tf_target_approach = ([-0.13951663,  0.01709836,  0.19042246],
                    [-0.02402769,  0.52414153, -0.01857238,  0.85108954])
tf_target_fitted = ([-0.13951663,  0.01709836,  0.17042246],
                    [-0.02402769,  0.52414153, -0.01857238,  0.85108954])

def generate_trajectory(tf_target_approach):
    tf_cur = S.p.getLinkState(env.robot, 23)[0:2]
    tf_target = S.p.getBasePositionAndOrientation(env.target)
    m_approach = np.dot(transform2homogeneousM(tf_target), transform2homogeneousM(tf_target_approach))
    tf_approach = homogeneousM2transform(m_approach)
    return interpolate_cartesian(tf_cur, tf_approach)

def goto_waypoint(tf_wp):
    q = S.p.calculateInverseKinematics(env.robot, 23, tf_wp[0], tf_wp[1])[:6]
    env.setJointValues(env.robot, env.armJoints, q)

def follow_trajectory(traj):
    for tf_wp in traj:
        goto_waypoint(tf_wp)
        sync()

def interpolate_cartesian(tf_cur, tf_goal):
    duration = 3.0
    dt = 0.1
    key_times = [0., duration]
    slerp = Slerp(key_times, R.from_quat([tf_cur[1], tf_goal[1]]))
    times = np.arange(0., duration, dt)
    interp = interp1d(key_times, np.array([tf_cur[0], tf_goal[0]]).transpose())
    return [(interp(tm), slerp(tm).as_quat()) for tm in times]

def transform2homogeneousM(tfobj):
    tfeul = tf.transformations.euler_from_quaternion(tfobj[1])
    tftrans = tfobj[0]
    tfobjM = tf.transformations.compose_matrix(angles=tfeul, translate=tftrans)
    return tfobjM

def homogeneousM2transform(tfobjM):
    scale, shear, angles, trans, persp = tf.transformations.decompose_matrix(tfobjM)
    quat = tf.transformations.quaternion_from_euler(*angles)
    return (trans, quat)

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


#sm = StateManager(mdl.time_window_size)
roi_hist = []
predicted_images = []

def reset(target_pose=None):

    env.resetRobot()
    if target_pose == None:
        # 'reaching_scene'
        # target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
        # target_ori = S.p.getQuaternionFromEuler([0,0,0])

        # 'reaching_2ways_scene'
        target_pos = np.append([0.0, -0.72] + [0.05, 0.05] * np.random.random(2), 0.79)
        target_ori = S.p.getQuaternionFromEuler(np.append([0,0], -1.0 + -0.5*np.random.random(1)))
    else:
        target_pos, target_ori = target_pose
    env.setObjectPosition(target_pos, target_ori)
    sync()
    
    img0 = captureRGB()
    js0 = env.getJointState()
    #js0 = normalize_joint_position(js0)
    #sm.initializeHistory(img0, js0)
    rec.clearFrames()

def captureRGB():
    img = cam.getImg()[2][:,:,:3]
    #img = cv2.resize(img, swap(mdl.input_image_size))
    return img / 255.

def captureSegImg():
    img = cam.getImg()[4]
    #img = cv2.resize(img, swap(mdl.input_image_size))
    return img / 255.

def getEFPos():
    s = S.p.getLinkState(env.robot, 7)
    return s[0]

def generate_eval_goal_acc_samples(n_samples):
    samples = []
    np.random.seed(0)
    for i in range(n_samples):
        target_pos = np.append([0.1, -0.65] + [0.1, 0.1] * np.random.random(2), 0.77)
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


    


                
                
# env.writeFrames(cam.getCameraConfig())
# $ convert -delay 20 -loop 0 *.jpg image_frames.gif
