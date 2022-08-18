import pybullet as p
import numpy as np
import pandas as pd
import yaml, os
import matplotlib.pyplot as plt

import forceGL3D

class CAMERA:
    def __init__(self, width=300, height=300, fov=50, near=0.1, far=2.0, shadow=True):
        self.cameraConfig = {}
        self.cameraConfig['shadow'] = shadow
        self.setProjectionMatrix(width, height, fov, near, far)
        self.setViewMatrix([0.0, -1.0, 2.0], [0.0, -0.9, 0.8], [0, 1, -1])
 
    def getImg(self):
        width,height = self.cameraConfig['imageSize']
        shadow = self.cameraConfig['shadow']
        return p.getCameraImage(width,
                                height,
                                self.view_matrix,
                                self.projection_matrix, 
                                lightDirection=[2,0,0], 
                                shadow=shadow,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL if shadow == True else p.ER_TINY_RENDERER)

    def setViewMatrix(self, eyePosition, targetPosition, upVector):
        self.cameraConfig['eyePosition'] = eyePosition
        self.cameraConfig['targetPosition'] = targetPosition
        self.cameraConfig['upVector'] = upVector
        self.view_matrix = p.computeViewMatrix(eyePosition, targetPosition, upVector)

    def setProjectionMatrix(self, width, height, fov, near, far):
        self.cameraConfig['imageSize'] = (width,height)
        self.cameraConfig['fov'] = fov
        self.cameraConfig['near'] = near
        self.cameraConfig['far'] = far
        aspect = width / height
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def getCameraConfig(self):
        return self.cameraConfig

class ForceCamera(CAMERA):
    def __init__(self):
        self.window_height = 240
        self.window_width = 320
        self.frc = forceGL3D.forceGL(self.window_width, self.window_height)
        # self.frc.setForceLimits()
        # copy values from the simulator
        self.frc.computeViewMatrix()
        self.frc.computeProjectionMatrixFOV()

    def getImg(self):
        self.frc.setForces()

    def setViewMatrix(self, eyePosition, targetPosition, upVector):
        super().setViewMatrix(eyePosition, targetPosition, upVector)
        self.frc.computeViewMatrix(eyePosition, targetPosition, upVector)
 
    def setProjectionMatrix(self, width, height, fov, near, far):
        super().setProjectionMatrix(width, height, fov, near, far)
        self.frc.computeProjectionMatrixFOV(fov, width / height, near, far)


class RECORDER:
    def __init__(self, cameraConfig):
        self.cameraConfig = cameraConfig
        self.groupNo = 1
        self.reset()

    def reset(self):
        self.previous_js = None
        self.clearFrames()
        
    def clearFrames(self):
        self.frameNo = 0
        self.frames = []

    def saveFrame(self, img, js, env, save_threshold=5e-2):
        if type(self.previous_js) != np.ndarray or np.linalg.norm(js - self.previous_js, ord=1) > save_threshold:
            print('save:[{}]: {}'.format(self.frameNo, js))
            d = {'frameNo':self.frameNo, 'jointPosition':js, 'image':img}
            for k,id in env.objects.items():
                d[k] = p.getBasePositionAndOrientation(id)
            self.frames.append(d)
            self.frameNo += 1
            self.previous_js = js

    def writeFrames(self):
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
        pd.to_pickle((self.cameraConfig, self.frames), os.path.join(group_dir, 'sim_states.pkl'))
        print('done')
        self.groupNo += 1
        self.reset()

class Environment:
    def __init__(self):
        pass

class SIM(Environment):
    def __init__(self, scene_file):
        super().__init__()
        self.rootdir = "../"
        self.connection = p.connect(p.GUI)
        p.resetSimulation()
        self.gravity = p.setGravity(0, 0, -9.81)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(self.rootdir)
        plane = p.loadURDF("specification/urdf/plane/plane.urdf")
        self.loadScene(scene_file)

    def loadScene(self, scene_file):
        with open(os.path.join(self.rootdir, 'specification/scenes', scene_file)) as f:
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
            cam = CAMERA(capture_size[1], capture_size[0], fov=fov, shadow=self.shadow)
            cam.setViewMatrix(*view_params)
            self.cameras[name] = cam

        robot_desc = scene_desc['robot']
        base_pose = robot_desc['base_position']
        self.robot = p.loadURDF(robot_desc['robot_model'], base_pose['xyz'], p.getQuaternionFromEuler(base_pose['rpy']), useFixedBase=True)
        self.armJoints = robot_desc['arm_joints']
        self.gripperJoints = robot_desc['gripper_joints']
        self.armInitialValues = scene_desc['robot']['initial_arm_pose']
        self.gripperInitialValues = scene_desc['robot']['initial_gripper_pose']

        d = scene_desc['environment'][0]
        self.cabinet = p.loadURDF(d['object'], d['xyz'], p.getQuaternionFromEuler(d['rpy']), useFixedBase=True, useMaximalCoordinates=True)
        self.objects = {}
        for d in scene_desc['objects']:
            self.objects[d['name']] = p.loadURDF(d['object'], d['xyz'], p.getQuaternionFromEuler(d['rpy']), useFixedBase=True, useMaximalCoordinates=True)
        self.target = self.objects.get('target')
            
        self.resetRobot()

    def getSceneDescription(self):
        return self.scene_desc
        
    def getCamera(self, name):
        return self.cameras[name]

    def setObjectPosition(self, xyz, rpy):
        p.resetBasePositionAndOrientation(self.target, xyz, rpy)
        
    def getTargetXYZ(self):
        pos, ori = p.getBasePositionAndOrientation(self.box)
        return np.array(pos)

    def getTCPXYZ(self):
        linkID = 7
        s = p.getLinkState(self.robot, linkID)
        # transform from wrist link to tool center point
        # w2t_tf = ([0.174 -0.015, 0.004, 0], p.getQuaternionFromEuler([0,0,0]))
        w2t_tf = ([0.174-0.03, 0, 0], p.getQuaternionFromEuler([0,0,0]))
        goalPos, goalOri = self.multiplyTransforms(s, w2t_tf)
        return np.array(goalPos)

    def setJointValues(self, robot, mask, values, mode=2, maxForce=500):
        arrayForces = [maxForce] * len(mask)
        p.setJointMotorControlArray(robot, mask, controlMode=mode, targetPositions=values, forces=arrayForces)

    def getArmJointPositions(self):
        return p.getJointStates(self.robot, self.armJoints)

    def setArmJointPositions(self, q):
        self.setJointValues(self.robot, self.armJoints, q)

    def getGripperJointPoisions(self):
        return p.getJointStates(self.robot, self.gripperJoints)

    def setGripperJointPositions(self, q):
        self.setJointValues(self.robot, self.gripperJoints, [q,-q,q,q,-q,q])   

    def moveEF(self, dl, da=None):
        s = p.getLinkState(self.robot, 7)
        # pos = s[0]
        # ori = s[1]
        # goalPos = np.array(pos) + dl
        # goalPos[2] = 0.835
        # goalOri = ori
        if (da is None):
            da = unit_quat()
        
        # goalPos, goalOri = self.multiplyTransforms(s, (dl, da))        
        goalPos, goalOri = self.multiplyTransforms((dl, da), s)
        goalPos = np.array(s[0]) + dl
        q = p.calculateInverseKinematics(self.robot, 7, goalPos, goalOri)[:6]
        self.setJointValues(self.robot, self.armJoints, q)

    def multiplyTransforms(self, t1, t2):
        return p.multiplyTransforms(t1[0], t1[1], t2[0], t2[1])
    
    def rotateEF(self, euler):
        linkID = 7
        s = p.getLinkState(self.robot, linkID)
        # transform from wrist link to tool center point
        w2t_tf = ([0.21,0,0], p.getQuaternionFromEuler([0,0,0]))
        d_tf = ([0,0,0], p.getQuaternionFromEuler(euler))
        goalPos, goalOri = self.multiplyTransforms(s, self.multiplyTransforms(self.multiplyTransforms(w2t_tf, d_tf), p.invertTransform(*w2t_tf)))
        # goalPos, goalOri = p.multiplyTransforms(pos, ori, [0,0,0], dq)
        q = p.calculateInverseKinematics(self.robot, linkID, goalPos, goalOri)[:6]
        self.setJointValues(self.robot, self.armJoints, q)     

    def getJointState(self, encode_gripper_state=True):
        js = p.getJointStates(self.robot, self.armJoints)
        armjv = list(zip(*js))[0]
        js = p.getJointStates(self.robot, self.gripperJoints)
        js = list(zip(*js))
        grpjv = js[0]
        grpforce = js[3]
        if encode_gripper_state:
            # gripperClosed = np.max(grpforce) > 1.0
            gripperClosed = grpjv[0] > 0.3
            return np.append(armjv, gripperClosed)
        else:
            return np.append(armjv, grpjv)
    
    def resetRobot(self):
        self.setArmJointPositions(self.armInitialValues)
        self.setGripperJointPositions(0.7)


class UI:
    def __init__(self):
        pass


class TASK:
    def __init__(self, env, ui):
        pass

    def reset(self):
        pass

    def teach(self):
        self.recorder = Recorder()

    def runTask(self, model):
        pass


        
# class VRController_ROI(VRController):
#     def __init__(self, isVR, use3Dmouse):
#         super().__init__(isVR)
#         self.use3Dmouse = use3Dmouse

#         if use3Dmouse:
#             self.last_msg = Twist()
#             self.last_joy_msg = Joy()
#             rospy.init_node("spacenav_receiver")
#             rospy.Subscriber("/spacenav/twist", Twist, self.spacenav_callback, queue_size=1)
#             rospy.Subscriber("/spacenav/joy", Joy, self.joy_callback, queue_size=1)
            
#     def spacenav_callback(self, msg):
#         self.last_msg = msg
#         # rospy.loginfo("%s", msg)

#     def joy_callback(self, msg):
#         self.last_joy_msg = msg

#     def update(self):
#         super().update()

#         self.B2 = 0
#         self.KTL = 0
#         self.KTR = 0
#         self.KTU = 0
#         self.KTD = 0
#         self.KTLU = 0
#         self.KTRU = 0
#         self.KTLD = 0
#         self.KTRD = 0
#         self.KU = 0
#         self.KD = 0
#         self.CG = 0
#         self.OG = 0
#         self.W = 0
#         self.BLK = 0
#         aa = p.getKeyboardEvents()
#         m = copy.copy(aa)
#         qTranslate = ord('t')
#         qRotate = ord('r')
#         qLeft = 65295
#         qRight = 65296
#         qUp = 65297
#         qDown = 65298
#         qReset = ord('q')
#         qWrite = ord('a')
#         qCloseG = ord('f')
#         qOpenG = ord('d')
#         qBreak = ord('b')
#         if qTranslate in m and m[qTranslate]&p.KEY_IS_DOWN: # change the gripper position
#             if qLeft in m:
#                 if qUp in m:
#                     self.KTLU = 1
#                 elif qDown in m:
#                     self.KTLD = 1
#                 else:
#                     self.KTL = 1
#             elif qRight in m:
#                 if qUp in m:
#                     self.KTRU = 1
#                 elif qDown in m:
#                     self.KTRD = 1
#                 else:
#                     self.KTR = 1
#             elif qUp in m:
#                 self.KTU = 1
#             elif qDown in m:
#                 self.KTD = 1
#         if qRotate in m and m[qRotate]&p.KEY_IS_DOWN: # change the gripper orientation
#             if qUp in m:
#                 self.KU = 1
#             elif qDown in m:
#                 self.KD = 1
#         if qReset in m and m[qReset]&p.KEY_WAS_TRIGGERED: # reset the environment
#             self.B2 = 1
#         if qWrite in m and m[qWrite]&p.KEY_WAS_TRIGGERED: # write the log to the files
#             self.W = 1
#         if qCloseG in m and m[qCloseG]&p.KEY_WAS_TRIGGERED: # close gripper
#             self.CG = 1
#         if qOpenG in m and m[qOpenG]&p.KEY_WAS_TRIGGERED: # open gripper
#             self.OG = 1
#         if qBreak in m and m[qBreak]&p.KEY_WAS_TRIGGERED: # break the control loop
#             self.BLK = 1


