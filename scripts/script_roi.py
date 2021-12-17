import SIM_ROI as S
import copy
import numpy as np
import math

capture_size = (180, 320)
# view_params = [0,-0.9,1.35], [0,-0.55,0.5], [0,0,1]
view_params = [0,-0.7,1.4], [0,-0.6,0.5], [0,0,1]

env = S.SIM_ROI(scene='reaching', is_vr=False)
control = S.VRController_ROI(0)
calc = S.CALC(env.ur5)
cam = S.CAMERA_ROI(capture_size[1], capture_size[0], fov=50)
cam.setViewMatrix(*view_params)

def loop():
    while True:
        img = cam.getImg()
        control.update()
        env.saveFrame(img)
        if control.B1:  #VR
            gripperOri = [1.75,0,1.57]
            gripperOriQ = calc.get_QfE(gripperOri)
            joy_pos = control.position
            val = calc.get_IK(env.ur5, 8,joy_pos, gripperOriQ)
            invJoints = [1,2,3,4,5,6,13,15,17,18,20,22]
            env.setJointValues(env.ur5,invJoints,val)
        if control.B2:  #VR
            env.setInitialPos()
        if control.W:
            env.writeFrames(cam.getCameraConfig())

        v = 0.012
        vx = vy = math.sqrt(v*v / 2.)
        vtheta = 3

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
        if control.KRL:
            env.rotateEF(np.deg2rad([0,0,vtheta]))
        if control.KRR:
            env.rotateEF(np.deg2rad([0,0,-vtheta]))
        if control.CG:
            env.closeGripper()
        if control.OG:
            env.openGripper()

loop()
