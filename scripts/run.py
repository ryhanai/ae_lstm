# -*- coding: utf-8 -*-

import roi_ae_lstm_v5 as roi
import SIM_ROI as S
import numpy as np
import cv2

trainer = roi.ROI_AE_LSTM_Trainer()
trainer.prepare_for_test(load_val_data=False)

width = 320
height = 180
time_window = 10

env = S.SIM_ROI(0)
control = S.VRController(0)
cam = S.CAMERA_ROI(width, height, fov=50)
cam.setViewMatrix([0,-0.9,1.35], [0,-0.55,0.5], [0,0,1])

S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(False)

def reset():
    env.setInitialPos()
    for i in range(100):
        S.p.stepSimulation()

def getImg():
    img = cam.getImg()[2][:,:,:3]
    img = cv2.resize(img, (160, 90))
    return img / 255.

def run(max_steps=20):
    img0 = getImg()
    js0 = env.getJointState()
    a = np.repeat(img0[None, :], time_window, axis=0)
    a = np.repeat(a[None, :], 32, axis=0)
    b = np.repeat(js0[None, :], time_window, axis=0)
    b = np.repeat(b[None, :], 32, axis=0)
    history = a, b
    for i in range(max_steps):
        print('i = ', i)
        img = getImg()
        js = env.getJointState()
        a = np.roll(history[0], -1)
        a[-1] = img
        b = np.roll(history[1], -1)
        b[-1] = js
        history = a,b
        imgs, jvs, rois = trainer.model.predict(history)
        jv = jvs[0]
        print(jv)
        env.moveArm(jv[:6])
        S.p.stepSimulation()
