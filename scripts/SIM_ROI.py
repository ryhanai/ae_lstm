pybullet_ur5_path = "/home/ryo/Program/moonshot/pybullet_ur5"

import sys
sys.path.append(pybullet_ur5_path)

from SIM import *
import numpy as np


class CAMERA_ROI(CAMERA):
    def __init__(self, width=300,height=300, fov=50, near=0.2, far=2.0):
        super().__init__(width, height, fov, near, far)

    def getImg(self):
        return p.getCameraImage(self.width,self.height,self.view_matrix,self.projection_matrix, lightDirection=[2,0,0], shadow=False, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    def setViewMatrix(self, eyePosition, targetPosition, upVector):
        self.view_matrix = p.computeViewMatrix(eyePosition, targetPosition, upVector)

class SIM_ROI(SIM):
    def __init__(self, is_vr=False):
        super().__init__(is_vr)
        self.armJoints = [1,2,3,4,5,6]
        self.gripperJoints = [13,15,17,18,20,22]

        p.removeBody(self.cabinet)
        p.setAdditionalSearchPath("../")
        self.cabinet = p.loadURDF("specification/urdf/objects/large_table.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
        p.removeBody(self.target)
        self.target = p.loadURDF("specification/urdf/objects/flat_target.urdf", [0.3,-0.6,0.77], useFixedBase=True)
        p.removeBody(self.box)
        self.box = p.loadURDF("specification/urdf/objects/box30_real.urdf", [0.3,-0.6,0.77])
        self.setInitialPos()
        self.groupNo = 493

    def setInitialPos(self):
        # pick&place
        # initialValues = [-0.0009408158538868226,
        #                  -0.9906506399912721,
        #                  1.5819908718696956,
        #                  -2.176581590986293,
        #                  -1.570484461359131,
        #                  0.0004259455658886804]
        # pushing
        armInitialValues = [-0.899109997666209,
                            -0.5263515248505674,
                            1.1386341089127756,
                            -0.820088253279429,
                            -0.7349581228169729,
                            0.1551369530256832]
        gripperInitialValues = [0, 0, 0, 0, 0, 0]
        self.setJointValues(self.ur5, self.armJoints, armInitialValues)
        self.setJointValues(self.ur5, self.gripperJoints, gripperInitialValues)

        box_pos = np.append([-0.2, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
        box_ori = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.box, box_pos, box_ori)
        target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.77)
        target_ori = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.target, target_pos, target_ori)

        self.previous_js = self.getJointState()
        self.frameNo = 0
        self.frames = []

    def moveArm(self, q):
        self.setJointValues(self.ur5, self.armJoints, q)
        
    def moveEF(self, d):
        s = p.getLinkState(self.ur5, 7)
        pos = s[0]
        ori = s[1]
        goalPos = np.array(pos) + d
        goalPos[2] = 0.835
        goalOri = ori
        q = p.calculateInverseKinematics(self.ur5, 7, goalPos, goalOri)[:6]
        self.setJointValues(self.ur5, self.armJoints, q)

    def multiplyTransforms(self, t1, t2):
        return p.multiplyTransforms(t1[0], t1[1], t2[0], t2[1])
        
    def rotateEF(self, euler):
        linkID = 7
        s = p.getLinkState(self.ur5, linkID)
        # transform from wrist link to tool center point
        w2t_tf = ([0.21,0,0], p.getQuaternionFromEuler([0,0,0]))
        d_tf = ([0,0,0], p.getQuaternionFromEuler(euler))
        goalPos, goalOri = self.multiplyTransforms(s, self.multiplyTransforms(self.multiplyTransforms(w2t_tf, d_tf), p.invertTransform(*w2t_tf)))
        # goalPos, goalOri = p.multiplyTransforms(pos, ori, [0,0,0], dq)
        q = p.calculateInverseKinematics(self.ur5, linkID, goalPos, goalOri)[:6]
        self.setJointValues(self.ur5, self.armJoints, q)

    def moveGripper(self, q):
        self.setJointValues(self.ur5, self.gripperJoints, [q,-q,q,q,-q,q])        
        
    def openGripper(self):
        #self.moveGripper(0)
        self.moveGripper(0.4)

    def closeGripper(self):
        self.moveGripper(0.8)

    def getJointState(self, encode_gripper_state=True):
        js = p.getJointStates(self.ur5, self.armJoints)
        armjv = list(zip(*js))[0]
        js = p.getJointStates(self.ur5, self.gripperJoints)
        js = list(zip(*js))
        grpjv = js[0]
        grpforce = js[3]
        if encode_gripper_state:
            # gripperClosed = np.max(grpforce) > 1.0
            gripperClosed = grpjv[0] > 0.3
            return np.append(armjv, gripperClosed)
        else:
            return np.append(armjv, grpjv)

    def saveFrame(self, img, save_threshold=5e-2):
        js = self.getJointState()
        if np.linalg.norm(js - self.previous_js, ord=1) > save_threshold:
            print('save:[{}]: {}'.format(self.frameNo, js))
            self.previous_js = js
            self.frames.append((self.frameNo, js, img))
            self.frameNo += 1

    def writeFrames(self):
        print('writing to files ...')
        _, joint_positions, images = zip(*self.frames)
        group_dir = 'data/{:d}'.format(self.groupNo)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        np.savetxt(os.path.join(group_dir, 'joint_position.txt'), joint_positions)
        for i,image in enumerate(images):
            w,h,rgb,depth,seg = image
            plt.imsave(os.path.join(group_dir, 'image_frame{:05d}.jpg'.format(i)), rgb)
            plt.imsave(os.path.join(group_dir, 'image_frame{:05d}_depth.jpg'.format(i)), depth, cmap=plt.cm.gray)
            plt.imsave(os.path.join(group_dir, 'image_frame{:05d}_seg.jpg'.format(i)), seg)
        print('done')
        self.groupNo += 1
        self.setInitialPos()
        
class VRController_ROI(VRController):
    def __init__(self, isVR):
        super().__init__(isVR)

    def update(self):
        super().update()

        self.B2 = 0
        self.KTL = 0
        self.KTR = 0
        self.KTU = 0
        self.KTD = 0
        self.KTLU = 0
        self.KTRU = 0
        self.KTLD = 0
        self.KTRD = 0
        self.KRL = 0
        self.KRR = 0
        self.CG = 0
        self.OG = 0
        self.W = 0
        aa = p.getKeyboardEvents()
        m = copy.copy(aa)
        qTranslate = ord('t')
        qRotate = ord('r')
        qLeft = 65295
        qRight = 65296
        qUp = 65297
        qDown = 65298
        qReset = ord('q')
        qWrite = ord('a')
        qCloseG = ord('f')
        qOpenG = ord('d')
        if qTranslate in m and m[qTranslate]&p.KEY_IS_DOWN: # change the gripper position
            if qLeft in m:
                if qUp in m:
                    self.KTLU = 1
                elif qDown in m:
                    self.KTLD = 1
                else:
                    self.KTL = 1
            elif qRight in m:
                if qUp in m:
                    self.KTRU = 1
                elif qDown in m:
                    self.KTRD = 1
                else:
                    self.KTR = 1
            elif qUp in m:
                self.KTU = 1
            elif qDown in m:
                self.KTD = 1
        if qRotate in m and m[qRotate]&p.KEY_IS_DOWN: # change the gripper orientation
            if qLeft in m:
                self.KRL = 1
            elif qRight in m:
                self.KRR = 1
        if qReset in m and m[qReset]&p.KEY_WAS_TRIGGERED: # reset the environment
            self.B2 = 1
        if qWrite in m and m[qWrite]&p.KEY_WAS_TRIGGERED: # write the log to the files
            self.W = 1
        if qCloseG in m and m[qCloseG]&p.KEY_WAS_TRIGGERED: # close gripper
            self.CG = 1
        if qOpenG in m and m[qOpenG]&p.KEY_WAS_TRIGGERED: # open gripper
            self.OG = 1

