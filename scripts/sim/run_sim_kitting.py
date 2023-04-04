# -*- coding: utf-8 -*-

import math
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pybullet_tools import *
import scipy.linalg
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


from core.utils import *
import SIM_KITTING as S


parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--scene', type=str, default='kitting_scene.yaml')
# parser.add_argument('-b', '--baseline', action='store_true')
parser.add_argument('-m', '--load_model', action='store_true')
parser.add_argument('-u', '--ui', type=str, default='none')
args = parser.parse_args()

message('scene = {}'.format(args.scene))
message('ui = {}'.format(args.ui))

if args.load_model:
    message('loading model ...')
    from attention import roi_cbam_v2 as mdl
    tr = mdl.prepare(task='test', dataset='kitting3')


env = S.SIM(scene_file=args.scene, rootdir='../../')
# S.p.changeDynamics(env.robot, 23, collisionMargin=0.0)
S.p.changeDynamics(env.target, 0, collisionMargin=-0.03)
S.p.changeDynamics(env.target, 1, collisionMargin=-0.03)

# remove shadow
S.p.configureDebugVisualizer(lightPosition=[0,0,10000])
S.p.configureDebugVisualizer(shadowMapIntensity=0.0)

# S.p.setCollisionFilterPair(env.robot, env.cabinet, 23, 0, 0)
cam = env.getCamera('camera1')
rec = S.RECORDER_KITTING(cam.getCameraConfig())

if args.ui == '3dmouse':
    import SpaceNavUI
    ui = SpaceNavUI.SpaceNavUI()


def eventLoop():
    while True:
        S.p.getMouseEvents()


def teach():
    # S.p.setTimeStep(1./240)
    S.p.setRealTimeSimulation(True)

    while True:
        img = cam.getImg()
        # control.update()
        js = env.getJointState(min_closed=0.7)
        rec.saveFrame(img, js, env)

        if ui.getEventSignal() == SpaceNavUI.UI_EV_LEFT_CLICK:
            env.setGripperJointPositions(0.65)
            release_object()

        if ui.getEventSignal() == SpaceNavUI.UI_EV_RIGHT_CLICK:
            rec.writeFrames()
            reset()
            continue

        v, w = ui.getControlSignal()
        w = S.p.getQuaternionFromEuler(w)
        env.moveEF(v, w)


def teach_procedurally(noise_approach=0.03):
    S.p.setRealTimeSimulation(True)
    # tf_target_approach_noise = (tf_target_approach[0] + noise_approach * (np.random.random(3) - 0.1), tf_target_approach[1])

    wps = gen_waypoints()
    follow_trajectory(generate_trajectory(wps[0], duration=3.5))
    follow_trajectory(generate_trajectory(wps[1], duration=0.8))
    follow_trajectory(generate_trajectory(wps[2], duration=1.5))
    follow_trajectory(generate_trajectory(wps[3], duration=0.8))
    release_object()
    env.setGripperJointPositions(0.65)
    start = time.time()
    while time.time() - start < 1.0:
        img = cam.getImg()
        js = env.getJointState(min_closed=0.7)
        rec.saveFrame(img, js, env, save_threshold=0.)
    follow_trajectory(generate_trajectory(wps[2], duration=0.8))
    rec.writeFrames()


def create_dataset(n):
    for i in range(n):
        reset()
        teach_procedurally()


# Goal State
# pen (robot, 23)
tf_approach = ((0.03664122521408607, -0.5315819169452275, 0.963800216862066),
               (0.25350660559819876,
                0.46699976144543603,
                -0.5618843155322776,
                0.6339807881054513))

# target
# tf_target = ((0.03975343991738431, -0.6731654428797486, 0.79),
#              (0.0, 0.0, -0.6185583387305793, 0.785738876209435))


def gen_waypoints():
    pen_id = 4
    tf_target = get_pose(env.target)
    tf_pen = get_pose(pen_id)
    tf_cur = S.p.getLinkState(env.robot, 11)[0:2]
    tf_grasp = multiply_transforms(invert_transform(tf_pen), tf_cur)

    # poses are defined with respect to the target object pose
    tf_pen3 = ((-0.01, 0, 0.02), quat_from_euler((0,0,np.pi)))  # pen pose fitted to the hole
    tf_pen_tip = ((0.073, 0, 0), unit_quat())

    tf_tip1 = multiply_transforms(tf_target, ((0.05,0,0.02), quat_from_euler((0,0.4, -0.3 + 0.6 * np.random.random()))))
    tf_tip0 = (np.array(tf_tip1[0]) + [0, 0, 0.01], tf_tip1[1])
    tf_tip2 = multiply_transforms(tf_target, ((0.064,0,0.02), quat_from_euler((0,0.3,0))))
    tf_tip3 = multiply_transforms(tf_target, ((0.064,0,0.02), quat_from_euler((0,0.06,0))))
    return [multiply_transforms(multiply_transforms(tf_tip, invert_transform(tf_pen_tip)), tf_grasp) for tf_tip in [tf_tip0, tf_tip1, tf_tip2, tf_tip3]]
    # return [multiply_transforms(tf_tip, invert_transform(tf_pen_tip)) for tf_tip in [tf_tip0, tf_tip1, tf_tip2, tf_tip3]]


# approach waypoint relative to tf_target
# tf_target_approach = ([-0.13951663,  0.01909836,  0.17542246],
#                     [-0.02402769,  0.52414153, -0.01857238,  0.85108954])
tf_target_approach = ([-0.14751663,  0.01109836,  0.19042246],
                    [-0.02402769,  0.52414153, -0.01857238,  0.85108954])
tf_target_fitted = ([-0.13951663,  0.01109836,  0.17042246],
                    [-0.02402769,  0.52414153, -0.01857238,  0.85108954])

# tf_hand_pen: <origin rpy="0 -1.0 0" xyz="0.19 0 0.05"/>


def generate_trajectory(tf_goal, duration=3.5):
    tf_cur = S.p.getLinkState(env.robot, 11)[0:2]
    return interpolate_cartesian(tf_cur, tf_goal, duration=duration)


def goto_waypoint(tf_wp):
    q = S.p.calculateInverseKinematics(env.robot, 11, tf_wp[0], tf_wp[1])[:6]
    env.setJointValues(env.robot, env.armJoints, q)

    
def follow_trajectory(traj):
    for tf_wp in traj:
        goto_waypoint(tf_wp)
        sync()
        img = cam.getImg()
        js = env.getJointState()
        rec.saveFrame(img, js, env)      


def interpolate_cartesian(tf_cur, tf_goal, duration=3.5):
    dt = 0.1
    key_times = [0., duration]
    slerp = Slerp(key_times, R.from_quat([tf_cur[1], tf_goal[1]]))
    times = np.arange(0., duration, dt)
    interp = interp1d(key_times, np.array([tf_cur[0], tf_goal[0]]).transpose())
    return [(interp(tm), slerp(tm).as_quat()) for tm in times]


# def transform2homogeneousM(tfobj):
#     tfeul = tf.transformations.euler_from_quaternion(tfobj[1])
#     tftrans = tfobj[0]
#     tfobjM = tf.transformations.compose_matrix(angles=tfeul, translate=tftrans)
#     return tfobjM


# def homogeneousM2transform(tfobjM):
#     scale, shear, angles, trans, persp = tf.transformations.decompose_matrix(tfobjM)
#     quat = tf.transformations.quaternion_from_euler(*angles)
#     return (trans, quat)


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


def grasp_object(name='pen'):
    # tf_hand = S.p.getLinkState(env.robot, 11)[0:2] # gripper_base_joint
    # m_hand = transform2homogeneousM(tf_hand)
    # m_hand_pen = transform2homogeneousM(tf_hand_obj)
    # m_pen = np.dot(m_hand, m_hand_pen)
    # tf_pen = homogeneousM2transform(m_pen)
    # env.setObjectPosition('pen', tf_pen[0], tf_pen[1])
    # sync()

    grasp_pos = np.array([0.19, 0, 0.05]) + np.array([0.019, 0, 0.005]) * (np.random.random(3) - 0.5)
    grasp_ori_euler = np.array([0, -1, 0]) + np.array([0, 0.2, 0]) * (np.random.random(3) - 0.5)
    tf_hand_obj=(grasp_pos, quat_from_euler(grasp_ori_euler))

    cstr = S.p.createConstraint(
        parentBodyUniqueId=env.robot,
        parentLinkIndex=11,
        childBodyUniqueId=env.objects[name],
        childLinkIndex=-1,
        jointType=S.p.JOINT_FIXED,
        jointAxis=[0,0,1],
        parentFramePosition=tf_hand_obj[0],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=tf_hand_obj[1]
    )
    return cstr


def release_object():
    cstr_ids = [S.p.getConstraintUniqueId(n) for n in range(S.p.getNumConstraints())]
    for cstr_id in cstr_ids:
        S.p.removeConstraint(cstr_id)


def reset(target_pose=None, relocate_target=True):
    env.resetRobot()
    sync()
    dp = np.append(np.array([0.0, -0.2]) + 0.2 * np.random.random(2), -0.03 + 0.06 * np.random.random())
    env.moveEF(dp)

    if relocate_target:
        if target_pose == None:
            # 'reaching_scene'
            # target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
            # target_ori = S.p.getQuaternionFromEuler([0,0,0])

            # 'reaching_2ways_scene'
            target_pos = np.append([0.0, -0.75] + 0.2 * np.random.random(2), 0.73)
            target_ori = S.p.getQuaternionFromEuler(np.append([0,0], -1.5 + 1.0*np.random.random(1)))
        else:
            target_pos, target_ori = target_pose
        env.setObjectPosition('target', target_pos, target_ori)

    env.setGripperJointPositions(0.0)
    grasp_object('pen')
    sync()
    env.setGripperJointPositions(0.72)
    sync()

    img0 = captureRGB()
    js0 = env.getJointState()
    rec.clearFrames()

    # parepare for motion generation
    js0 = normalize_joint_position(js0)
    sm.initializeHistory(img0, js0)
    roi_history = []
    attention_map_history = []


def captureRGB():
    img = cam.getImg()[2][:,:,:3]
    img = cv2.resize(img, swap(mdl.input_image_size))
    return img / 255.


def captureSegImg():
    img = cam.getImg()[4]
    # img = cv2.resize(img, swap(mdl.input_image_size))
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
        results.append((p1, p2))
    return results


def calc_train_data_bias(groups=range(1, 5)):
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
        reset()
        for s in pd.read_pickle('~/Dataset/dataset2/{}/{}/sim_states.pkl'.format(task, g))[1]:
            q = s['jointPosition']
            name = 'target'
            p_target = s[name]
            env.setObjectPosition(name, p_target[0], p_target[1])
            env.setArmJointPositions(q[:6])
            if q[6] < 0.5:
                env.setGripperJointPositions(0.65)
                release_object()
            sync()
            js = env.getJointState()
            img = cam.getImg()
            rec.saveFrame(img, js, env)
        rec.writeFrames()



def normalize_joint_position(q):
    return tr.val_ds.normalize_joint_position(q)


def unnormalize_joint_position(q):
    return tr.val_ds.unnormalize_joint_position(q)


n_runs = 0
sm = StateManager(mdl.time_window_size)
roi_history = []
attention_map_history = []


def run(max_steps=50, anim_gif=False):
    global n_runs
    attention_map_history = []

    for i in range(max_steps):
        print('i = ', i)
        img = captureRGB()
        js = env.getJointState()

        js = normalize_joint_position(js)
        sm.addState(img, js)
        y_pred = tr.predict(sm.getHistory())
        feat_pred, q_pred, attention_map = y_pred
        input_image = sm.getHistory()[0][:,-1]
        attention_map_history.append(tr.post_process(input_image, y_pred))

        jv = q_pred[0]
        jv = unnormalize_joint_position(jv)

        print(jv)
        env.setArmJointPositions(jv[:6])
        if jv[6] < 0.5:
            release_object()
        sync()

    if anim_gif:
        create_anim_gif_from_images(attention_map_history, out_filename='run_{:05d}.gif'.format(n_runs))
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
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    coordinate(ax, [-0.6, 0.6], [-1.0, -0.2])
    for group in groups:
        cs, ps, ls = predict_sequence_open(group)
        draw_predictions_and_labels(ax, cs, ps, ls)
    plt.show()


# env.writeFrames(cam.getCameraConfig())
# $ convert -delay 20 -loop 0 *.jpg image_frames.gif
