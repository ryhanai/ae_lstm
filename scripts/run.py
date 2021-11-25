# -*- coding: utf-8 -*-

import roi_ae_lstm_v5 as roi
import SIM_ROI as S
import numpy as np
import cv2
import matplotlib.pyplot as plt

trainer = roi.ROI_AE_LSTM_Trainer()
trainer.prepare_for_test(load_val_data=False)

width = 320
height = 180
time_window = 10
view_params = ([0,-0.9,1.35], [0,-0.55,0.5], [0,0,1])

env = S.SIM_ROI(0)
control = S.VRController(0)
cam = S.CAMERA_ROI(width, height, fov=50)
cam.setViewMatrix(*view_params)

#S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(False)

def reset():
    env.setInitialPos()
    for i in range(100):
        S.p.stepSimulation()

def getImg():
    img = cam.getImg()[2][:,:,:3]
    img = cv2.resize(img, (160, 90))
    return img / 255.

def getSegImg():
    img = cam.getImg()[4]
    img = cv2.resize(img, (160, 90))
    return img / 255.

jmin, jmax = roi.joint_position_range()
def normalize_joint_position(q):
    return (q - jmin) / (jmax - jmin)
def unnormalize_joint_position(q):
    return q * (jmax - jmin) + jmin

def initialize():
    img0 = getImg()
    js0 = env.getJointState()
    a = np.repeat(img0[None, :], time_window, axis=0)
    a = np.repeat(a[None, :], 32, axis=0)
    b = np.repeat(js0[None, :], time_window, axis=0)
    b = np.repeat(b[None, :], 32, axis=0)
    history = a, b
    return history

def initialize2():
    trainer.prepare_for_test()
    val_data = trainer.val_ds[0]
    a = np.array(val_data[1][:time_window])
    a = np.repeat(a[None, :], 32, axis=0)
    b = val_data[0][:time_window]
    b = np.repeat(b[None, :], 32, axis=0)
    history = a, b
    return history

history = initialize2()

def roll(history, img, js):
    a = np.roll(history[0], -1, axis=1)
    a[:,-1] = img
    b = np.roll(history[1], -1, axis=1)
    b[:,-1] = js
    return a,b

def getEFPos():
    s = S.p.getLinkState(env.ur5, 7)
    return s[0]

def save():
    global history
    img = getImg()
    js = env.getJointState()
    js = normalize_joint_position(js)
    history = roll(history, img, js)

def run(max_steps=50, anim_gif=False):
    global history

    frames = []
    for i in range(max_steps):
        print('i = ', i)
        img = getImg()
        frames.append(img)
        js = env.getJointState()
        p_before = getEFPos()
        js = normalize_joint_position(js)
        history = roll(history, img, js)
        imgs, jvs, rois = trainer.model.predict(history)
        jv = jvs[0]
        jv = unnormalize_joint_position(jv)
        print(jv)
        env.moveArm(jv[:6])
        for j in range(10):
            S.p.stepSimulation()

    if anim_gif:
        roi.create_anim_gif(frames, 'run.gif')

def sync(steps=100):
    for j in range(steps):
        S.p.stepSimulation()

def coordinate(axes, range_x, range_y, grid = True,
               xyline = True, xlabel = 'x', ylabel = 'y'):
    axes.set_xlabel(xlabel, fontsize = 16)
    axes.set_ylabel(ylabel, fontsize = 16)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    if grid == True:
        axes.grid()
    if xyline == True:
        axes.axhline(0, color = 'gray')
        axes.axvline(0, color = 'gray')

def draw_vector(axes, loc, vector, color='red', scale=1, width=1):
    axes.quiver(loc[0], loc[1],
                vector[0], vector[1], color = color,
                angles='xy', scale_units='xy', scale=scale, width=width)

def draw_trajectory(ax, traj, color='red', scale=0.4, width=0.002):
    for p1,p2 in traj:
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        draw_vector(ax, p1, p2-p1, color=color, scale=scale, width=width)

def draw_predictions_and_labels(ax, cs, ps, ls, colors=['red','blue'], scale=0.4, width=0.002):
    trajs = [zip(cs[::3],ls[::3]), zip(cs[::3],ps[::3])]
    for traj,color in zip(trajs, colors):
        draw_trajectory(ax, traj, color=color, scale=scale, width=width)

def predict_for_group(group_num=0, interval=3):
    res = []
    traj = trainer.process_sequence(group_num)
    for c,p,l in traj:
        c = unnormalize_joint_position(c)
        env.moveArm(c[:6])
        sync()
        pos_c = getEFPos()
        p = unnormalize_joint_position(p)
        env.moveArm(p[:6])
        sync()
        pos_p = getEFPos()
        l = unnormalize_joint_position(l)
        env.moveArm(l[:6])
        sync()
        pos_l = getEFPos()
        wp = (pos_c, pos_p, pos_l)
        print(wp)
        res.append(wp)

    cs,ps,ls = zip(*res)
    cs = np.array(cs)
    ps = np.array(ps)
    ls = np.array(ls)
    return cs,ps,ls

def visualize_predicted_vectors(groups=range(10)):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    coordinate(ax, [-0.6,0.6],[-1.0,-0.2])
    for group in groups:
        cs,ps,ls = predict_for_group(group)
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

# x,y,t,_ = trainer.test()
# def f(i):
#     env.moveArm(unnormalize_joint_position(x[i][-1])[:-1])
#     sync()
#     ximg = getImg()
#     env.moveArm(unnormalize_joint_position(t[i])[:-1])
#     sync()
#     yimg = getImg()

#     fig = plt.figure(figsize=(10,samples))
#     fig.subplots_adjust(hspace=0.1)

#     for p in range(samples):
#         ax = fig.add_subplot(samples//4, 4, p+1)
#         ax.axis('off')
#         ax.imshow(images[p])
#         if len(rois) > 0:
#             roi = rois[samples][0]
#             height = images[p].shape[0]
#             width = images[p].shape[1]
#             x = width * roi[1]
#             w = width * (roi[3] - roi[1])
#             y = height * roi[0]
#             h = height * (roi[2] - roi[0])
#             rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='red', fill=False) # x,y,w,h [pixels]
#             ax.add_patch(rect)


# from PIL import Image

# def overlay(img1, img2):
#     img1 = img1 * 255
#     im1 = Image.fromarray(img1.astype(np.uint8))
#     img2 = img2 * 255
#     im2 = Image.fromarray(img2.astype(np.uint8))
#     im2.putalpha(128)
#     merged = Image.new('RGBA', (160,90), (0,0,0,0))
#     merged.paste(im1, (0,0), None)
#     merged.paste(im2, (0,0), None)
#     plt.imshow(merged)
#     plt.show()
#     return merged
