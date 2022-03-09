from SIM import *
import numpy as np
import pandas as pd
import yaml


class CAMERA_ROI(CAMERA):
    def __init__(self, width=300, height=300, fov=50, near=0.2, far=2.0, shadow=True):
        super().__init__(width, height, fov, near, far)
        self.cameraConfig = {}
        self.cameraConfig['imageSize'] = (width,height)
        self.cameraConfig['fov'] = fov
        self.cameraConfig['near'] = near
        self.cameraConfig['far'] = far
        self.shadow = shadow

    def getImg(self):
        return p.getCameraImage(self.width,self.height,self.view_matrix,self.projection_matrix, lightDirection=[2,0,0], shadow=self.shadow, renderer=p.ER_BULLET_HARDWARE_OPENGL if self.shadow == True else p.ER_TINY_RENDERER)

    def setViewMatrix(self, eyePosition, targetPosition, upVector):
        self.cameraConfig['eyePosition'] = eyePosition
        self.cameraConfig['targetPosition'] = targetPosition
        self.cameraConfig['upVector'] = upVector
        self.view_matrix = p.computeViewMatrix(eyePosition, targetPosition, upVector)

    def getCameraConfig(self):
        return self.cameraConfig

class SIM_ROI(SIM):
    def __init__(self, scene_file, is_vr=False):
        super().__init__(is_vr)
        self.armJoints = [1,2,3,4,5,6]
        self.gripperJoints = [13,15,17,18,20,22]

        p.removeBody(self.cabinet)
        p.setAdditionalSearchPath("../")
        self.loadScene(scene_file)
        self.groupNo = 1

    def loadScene(self, scene_file):
        p.removeBody(self.target)
        p.removeBody(self.box)

        with open(os.path.join('../specification/scenes', scene_file)) as f:
            scene_desc = yaml.safe_load(f)

        self.scene_desc = scene_desc
            
        self.task = scene_desc['task']
        self.shadow = scene_desc['rendering']['shadow']

        self.cameras = {}
        for cam_desc in scene_desc['cameras']:
            name = cam_desc['name']
            view_params = cam_desc['view_params']
            fov = cam_desc['fov']
            capture_size = cam_desc['capture_size']
            cam = CAMERA_ROI(capture_size[1], capture_size[0], fov=fov, shadow=self.shadow)
            cam.setViewMatrix(*view_params)
            self.cameras[name] = cam
        
        self.armInitialValues = scene_desc['robot']['initial_arm_pose']
        self.gripperInitialValues = scene_desc['robot']['initial_gripper_pose']
        
        d = scene_desc['environment'][0]
        self.cabinet = p.loadURDF(d['object'], d['xyz'], p.getQuaternionFromEuler(d['rpy']), useFixedBase=True)

        self.objects = {}
        for d in scene_desc['objects']:
            self.objects[d['name']] = p.loadURDF(d['object'], d['xyz'], p.getQuaternionFromEuler(d['rpy']), useFixedBase=True)
            
        if self.task == 'reaching':
            self.target = self.objects['target']
            
        # if self.task == 'pushing':
        #     self.target = p.loadURDF("specification/urdf/objects/flat_target.urdf", [0.3,-0.6,0.77], useFixedBase=True)
        #     self.box = p.loadURDF("specification/urdf/objects/box30_real.urdf", [0.3,-0.6,0.77])
        #     self.objects = {'target':self.target, 'box':self.box}
            
        self.resetRobot()
        self.clearFrames()

    def getSceneDescription(self):
        return self.scene_desc
        
    def getCamera(self, name):
        return self.cameras[name]
    
    def resetRobot(self):
        self.moveArm(self.armInitialValues)
        self.openGripper()
        self.previous_js = self.getJointState()

    def setObjectPosition(self, xyz, rpy):
        # if self.task == 'pushing':
        #     if len(positions) == 2:
        #         box_pos = positions[0]
        #         target_pos = positions[1]
        #     else:
        #         box_pos = np.append([-0.2, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
        #         target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.77)
        #     box_ori = p.getQuaternionFromEuler([0,0,0])
        #     target_ori = p.getQuaternionFromEuler([0,0,0])
        #     p.resetBasePositionAndOrientation(self.box, box_pos, box_ori)

        p.resetBasePositionAndOrientation(self.target, xyz, rpy)

    # def setInitialPos(self):
    #     self.resetRobot()
    #     target_pos = np.append([0.1, -0.75] + [0.2, 0.3] * np.random.random(2), 0.79)
    #     target_ori = p.getQuaternionFromEuler([0,0,0])
    #     self.setObjectPosition(target_pos, target_ori)

    def clearFrames(self):
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

