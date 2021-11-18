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

env = S.SIM_ROI(0)
control = S.VRController(0)
cam = S.CAMERA_ROI(width, height, fov=50)
cam.setViewMatrix([0,-0.9,1.35], [0,-0.55,0.5], [0,0,1])

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
    a[-1] = img
    a = history[0]
    b = np.roll(history[1], -1, axis=1)
    b[-1] = js
    return a,b

def run(max_steps=20):
    global history

    for i in range(max_steps):
        print('i = ', i)
        img = getImg()
        js = env.getJointState()
        js = normalize_joint_position(js)
        history = roll(history, img, js)
        imgs, jvs, rois = trainer.model.predict(history)
        jvs = unnormalize_joint_position(jvs)
        jv = jvs[0]
        print(jv)
        env.moveArm(jv[:6])
        for j in range(1):
            S.p.stepSimulation()
        #var = input('>')


# x,y,t,_ = trainer.test()
# def f(i):
#     def sync():
#         for j in range(100):
#             S.p.stepSimulation()

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
