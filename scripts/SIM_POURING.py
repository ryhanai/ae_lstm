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

        pose = scene_desc['robot']['base_position']
        p.resetBasePositionAndOrientation(self.ur5, pose['xyz'], p.getQuaternionFromEuler(pose['rpy']))
        
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
        # w2t_tf = ([0.174 -0.015, 0.004, 0], p.getQuaternionFromEuler([0,0,0]))
        w2t_tf = ([0.174-0.03, 0, 0], p.getQuaternionFromEuler([0,0,0]))
        goalPos, goalOri = self.multiplyTransforms(s, w2t_tf)
        return np.array(goalPos)
    
    def moveArm(self, q):
        self.setJointValues(self.ur5, self.armJoints, q)
        
    def moveEF(self, dl, da=None):
        s = p.getLinkState(self.ur5, 7)
        # pos = s[0]
        # ori = s[1]
        # goalPos = np.array(pos) + dl
        # goalPos[2] = 0.835
        # goalOri = ori
        if (da is None):
            da = unit_quat()
        
        goalPos, goalOri = self.multiplyTransforms(s, (dl, da))
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


import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
        
class VRController_ROI(VRController):
    def __init__(self, isVR, use3Dmouse):
        super().__init__(isVR)
        self.use3Dmouse = use3Dmouse

        if use3Dmouse:
            self.last_msg = Twist()
            rospy.init_node("spacenav_receiver")
            rospy.Subscriber("/spacenav/twist", Twist, self.spacenav_callback)

    def spacenav_callback(self, msg):
        self.last_msg = msg
        # rospy.loginfo("%s", msg)
        
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
        self.BLK = 0
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
        qBreak = ord('b')
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
        if qBreak in m and m[qBreak]&p.KEY_WAS_TRIGGERED: # break the control loop
            self.BLK = 1


# class VRController_ROI(VRController):
#     def __init__(self, isVR):
#         super().__init__(isVR)

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
#         self.KRL = 0
#         self.KRR = 0
#         self.CG = 0
#         self.OG = 0
#         self.W = 0
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
#             if qLeft in m:
#                 self.KRL = 1
#             elif qRight in m:
#                 self.KRR = 1
#         if qReset in m and m[qReset]&p.KEY_WAS_TRIGGERED: # reset the environment
#             self.B2 = 1
#         if qWrite in m and m[qWrite]&p.KEY_WAS_TRIGGERED: # write the log to the files
#             self.W = 1
#         if qCloseG in m and m[qCloseG]&p.KEY_WAS_TRIGGERED: # close gripper
#             self.CG = 1
#         if qOpenG in m and m[qOpenG]&p.KEY_WAS_TRIGGERED: # open gripper
#             self.OG = 1

from collections import namedtuple

def Point(x=0, y=0, z=0):
    return np.array([x, y, z])

def get_pose(body):
    return p.getBasePositionAndOrientation(body)

def get_point(body):
    return get_pose(body)[0]

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_euler(body):
    return euler_from_quat(get_quat(body))

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_euler(body, euler):
    set_quat(body, quat_from_euler(euler))

def set_position(body, x=None, y=None, z=None):
    # TODO: get_position
    position = list(get_point(body))
    for i, v in enumerate([x, y, z]):
        if v is not None:
            position[i] = v
    set_point(body, position)
    return position

def set_orientation(body, roll=None, pitch=None, yaw=None):
    orientation = list(get_euler(body))
    for i, v in enumerate([roll, pitch, yaw]):
        if v is not None:
            orientation[i] = v
    set_euler(body, orientation)
    return orientation


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler) # TODO: extrinsic (static) vs intrinsic (rotating)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat) # rotation around fixed axis

def intrinsic_euler_from_quat(quat):
    #axes = 'sxyz' if static else 'rxyz'
    return euler_from_quaternion(quat, axes='rxyz')

def unit_point():
    return (0., 0., 0.)

def unit_quat():
    return quat_from_euler([0, 0, 0]) # [X,Y,Z,W]

def quat_from_axis_angle(axis, angle): # axis-angle
    #return get_unit_vector(np.append(vec, [angle]))
    return quaternion_about_axis(angle, axis)
    #return np.append(math.sin(angle/2) * get_unit_vector(axis), [math.cos(angle / 2)])

def unit_pose():
    return (unit_point(), unit_quat())

NULL_ID = -1
STATIC_MASS = 0

RGB = namedtuple('RGB', ['red', 'green', 'blue'])
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
MAX_RGB = 2**8 - 1

RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)

def create_collision_shape(geometry, pose=unit_pose()):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        #'physicsClientId': CLIENT,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)

def create_visual_shape(geometry, pose=unit_pose(), color=RED, specular=None):
    if (color is None): # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        #'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) # if collision else NULL_ID
    return collision_id, visual_id

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id)

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def create_sphere(radius, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_sphere_geometry(radius), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def fill_water(num_droplets = 80, pos=np.array([-0.14, -0.63, 0.9]), radius=0.002):
    # [-0.14, -0.63, 1.02]
    # [0.27, -0.65, 0.9] glass
    droplets = [create_sphere(radius, mass=0.01) for _ in range(num_droplets)] # kg
    
    cup_thickness = 0.001
    # lower, upper = get_lower_upper(cup)
    # buffer = cup_thickness + radius
    # lower = np.array(lower) + buffer*np.ones(len(lower))
    # upper = np.array(upper) - buffer*np.ones(len(upper))
    # limits = zip(lower, upper)
    # x_range, y_range = limits[:2]

    # for droplet in droplets:
    #     x = np.random.uniform(*x_range)
    #     y = np.random.uniform(*y_range)
    #     set_point(droplet, Point(x,y,z))

    for i, droplet in enumerate(droplets):
        x, y = pos[:2] + np.random.normal(0, 1e-3, 2)
        z = pos[2] + i*(2*radius+1e-3)
        set_point(droplet, Point(x, y, z))

    # enable_gravity()
    
