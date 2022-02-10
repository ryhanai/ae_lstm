from SIM import *
import numpy as np
import pandas as pd


class CAMERA_ROI(CAMERA):
    def __init__(self, width=300, height=300, fov=40, near=0.2, far=2.0):
        super().__init__(width, height, fov, near, far)
        self.cameraConfig = {}
        self.cameraConfig['imageSize'] = (width,height)
        self.cameraConfig['fov'] = fov
        self.cameraConfig['near'] = near
        self.cameraConfig['far'] = far

    def getImg(self):
        return p.getCameraImage(self.width,self.height,self.view_matrix,self.projection_matrix, lightDirection=[2,0,0], shadow=False, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    def setViewMatrix(self, eyePosition, targetPosition, upVector):
        self.cameraConfig['eyePosition'] = eyePosition
        self.cameraConfig['targetPosition'] = targetPosition
        self.cameraConfig['upVector'] = upVector
        self.view_matrix = p.computeViewMatrix(eyePosition, targetPosition, upVector)

    def getCameraConfig(self):
        return self.cameraConfig

class SIM_ROI(SIM):
    def __init__(self, scene='pushing', is_vr=False):
        #super().__init__(is_vr)

        if is_vr:
            self.connection = p.connect(p.SHARED_MEMORY)        
        else:
            self.connection = p.connect(p.GUI)

        p.resetSimulation()
        self.gravity = p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ##
        pybullet_ur5_path = "/home/rspuser/Program/moonshot-icps/pybullet_ur5"
        p.setAdditionalSearchPath(pybullet_ur5_path)
        plane = p.loadURDF("specification/urdf/plane/plane.urdf")

        self.armJoints = [1,2,3,4,5,6]
        self.gripperJoints = [13,15,17,18,20,22]
        #self.armInitialValues = [0, -1.179, 1.526, -1.948, -1.571, 0]
        self.armInitialValues = [-0.20579837836525972,
                                 -1.206177708473775,
                                 1.5616389852747876,
                                 -1.9547686737957353,
                                 -1.564884914562381,
                                 -0.20555815954683102]
        
        p.setAdditionalSearchPath("../")
        ur5_pos = [0,0,0.795]
        ur5_ori = p.getQuaternionFromEuler([0,0,-1.57])
        self.ur5 = p.loadURDF("specification/urdf/robots/xxx.urdf", ur5_pos, ur5_ori, useFixedBase=True)
        
        self.cabinet = p.loadURDF("specification/urdf/objects/large_table.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
        self.scene = scene
        self.loadScene()

        self.groupNo = 1
        realSim = p.setRealTimeSimulation(True)  

    def loadScene(self):
        # p.removeBody(self.target)
        # p.removeBody(self.box)

        self.target = p.loadURDF("specification/urdf/objects/hole_object.urdf", [0.3,-0.6,0.795], useFixedBase=True)
        self.objects = {'target':self.target}

        self.max_force = 200
        self.max_velocity = 0.35
        #self.num_arm_joints = 6

        # Apply the joint positions
        for i, jointIndex in enumerate(self.armJoints):
            p.resetJointState(self.ur5, jointIndex, self.armInitialValues[i])
            p.setJointMotorControl2(self.ur5, jointIndex, p.POSITION_CONTROL,
                                    targetPosition=self.armInitialValues[i], force=self.max_force,
                                    maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)

        
        self.setInitialPos()

    def resetRobot(self):
        self.moveArm(self.armInitialValues)
        self.moveGripper(0.47)

        self.previous_js = self.getJointState()

    def setObjectPosition(self, *positions):
        if len(positions) == 1:
            target_pos = positions[0]
        else:
            target_pos = np.append([0.0, -0.8] + [0.5, 0.3] * np.random.random(2), 0.795)
        target_ori = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.target, target_pos, target_ori)
         
    def setInitialPos(self):
        self.resetRobot()
        self.setObjectPosition()
        self.frameNo = 0
        self.frames = []

    def getTargetXYZ(self):
        pos, ori = p.getBasePositionAndOrientation(self.box)
        return np.array(pos)

    def getTCPXYZ(self):
        linkID = 7
        s = p.getLinkState(self.ur5, linkID)
        # transform from wrist link to tool center point
        w2t_tf = ([0.174 -0.015, 0.004, 0], p.getQuaternionFromEuler([0,0,0]))
        goalPos, goalOri = self.multiplyTransforms(s, w2t_tf)
        return np.array(goalPos)
    
    def moveArm(self, q):
        self.setJointValues(self.ur5, self.armJoints, q)
        
    def moveEF(self, d):
        js = p.getJointStates(self.ur5, self.armJoints)
        jvels = np.array(list(zip(*js))[1])
        if not np.allclose(jvels, np.zeros(6), atol=5e-2):
            return
        s = p.getLinkState(self.ur5, 7)
        pos = s[0]
        ori = s[1]
        goalPos = np.array(pos) + d
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
        self.moveGripper(0)

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
            d = {'frameNo':self.frameNo, 'jointPosition':js, 'image':img}
            for k,id in self.objects.items():
                d[k] = p.getBasePositionAndOrientation(id)
            self.frames.append(d)
            self.frameNo += 1
            self.previous_js = js

    def writeFrames(self, cameraConfig):
        print('writing to files ...')
        group_dir = 'data/{:d}'.format(self.groupNo)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        joint_positions = []
        for frame in self.frames:
            joint_positions.append(frame['jointPosition'])
            frameNo = frame['frameNo']
            w,h,rgb,depth,seg = frame['image']
            plt.imsave(os.path.join(group_dir, 'image_frame{:05d}.jpg'.format(frameNo)), rgb)
            # plt.imsave(os.path.join(group_dir, 'image_frame{:05d}_depth.jpg'.format(frameNo)), depth, cmap=plt.cm.gray)
            # plt.imsave(os.path.join(group_dir, 'image_frame{:05d}_seg.jpg'.format(frameNo)), seg)

        np.savetxt(os.path.join(group_dir, 'joint_position.txt'), joint_positions)

        for f in self.frames:
            f.pop('image')
        pd.to_pickle((cameraConfig, self.frames), os.path.join(group_dir, 'sim_states.pkl'))
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
        self.KU = 0
        self.KD = 0
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
            if qUp in m:
                self.KU = 1
            elif qDown in m:
                self.KD = 1
        if qReset in m and m[qReset]&p.KEY_WAS_TRIGGERED: # reset the environment
            self.B2 = 1
        if qWrite in m and m[qWrite]&p.KEY_WAS_TRIGGERED: # write the log to the files
            self.W = 1
        if qCloseG in m and m[qCloseG]&p.KEY_WAS_TRIGGERED: # close gripper
            self.CG = 1
        if qOpenG in m and m[qOpenG]&p.KEY_WAS_TRIGGERED: # open gripper
            self.OG = 1

