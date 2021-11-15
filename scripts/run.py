# -*- coding: utf-8 -*-

import roi_ae_lstm_v5 as roi
import SIM_ROI as S
import numpy as np

# trainer = roi.ROI_AE_LSTM_Trainer()
# trainer.prepare_for_test(load_val_data=False)

width = 320
height = 180
time_window = 10

env = S.SIM_ROI(0)
control = S.VRController(0)
cam = S.CAMERA_ROI(width, height, fov=50)
cam.setViewMatrix([0,-0.9,1.35], [0,-0.55,0.5], [0,0,1])

S.p.setTimeStep(1./240)
realSim = S.p.setRealTimeSimulation(False)

def run(max_steps=20):
    img0 = cam.getImg()
    js0 = env.getJointState()
    history = np.tile(img0, (1, time_window, 1)), np.tile(js0, (1, time_window, 1))
    
    for i in range(max_steps):
        img = cam.getImg()
        js = env.getJointState()
        a = np.roll(history[0], -1)
        a[-1] = img
        b = np.roll(history[1], -1)
        b[-1] = js
        history = a,b
        #img, jvref = trainer.model.predict(history)
        #print(jvref)l
        print('i = ', i)
        for j in range(8):
            S.p.stepSimulation()
        #env.control(jvref)
