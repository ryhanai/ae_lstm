# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, Joy, JointState
from trajectory_msgs.msg import JointTrajectory
from jog_msgs.msg import JogFrame
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf2_ros
import numpy as np
import message_filters
import os
import pandas as pd
from tf.transformations import *

bridge = CvBridge()
groupNo = 1
frameNo = 0
frames = []
recording_now = False

def state_callback(image_msg, joint_msg, traj_msg):
    global frameNo
    global frames
    global recording_now
    #print(image_msg.header.stamp, joint_msg.header.stamp, traj_msg.header.stamp)
    if recording_now:
        frames.append([image_msg, joint_msg, traj_msg])

def start_recording():
    global frameNo
    global frames
    global recording_now
    print('start recording')
    frameNo = 0
    frames = []
    recording_now = True
    
def stop_recording():
    global groupNo
    global recording_now
    global frames
    recording_now = False
    print('stop recording')
    print('writing to files ...')
    group_dir = 'data/{:d}'.format(groupNo)
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)

    output = []
        
    for i,frame in enumerate(frames):
        try:
            cv2_img = bridge.imgmsg_to_cv2(frame[0], "bgr8")
            # img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('image', img)
            # cv2.waitKey(1)
        except CvBridgeError as err:
            print(err)
            return
        else:
            cv2.imwrite(os.path.join(group_dir, 'image_frame{:05d}.jpg'.format(i)), cv2_img)
            msg = Image()
            msg.header = frame[0].header
            msg.height = frame[0].height
            msg.width = frame[0].width
            msg.encoding = frame[0].encoding
            msg.is_bigendian = frame[0].is_bigendian
            msg.step = frame[0].step
            output.append([msg,frame[1],frame[2]])

    pd.to_pickle(output, os.path.join(group_dir, 'states.pkl'))

    print('done')
    groupNo += 1
        
    # joint_positions = []
    # for frame in self.frames:
    #     joint_positions.append(frame['jointPosition'])
    #     frameNo = frame['frameNo']
    #     w,h,rgb,depth,seg = frame['image']
    #     plt.imsave(os.path.join(group_dir, 'image_frame{:05d}.jpg'.format(frameNo)), rgb)

    # np.savetxt(os.path.join(group_dir, 'joint_position.txt'), joint_positions)

    # for f in self.frames:
    #     f.pop('image')
    # pd.to_pickle((cameraConfig, self.frames), os.path.join(group_dir, 'sim_states.pkl'

# def saveFrame(self, img, save_threshold=5e-2):
#     js = self.getJointState()
#     if np.linalg.norm(js - self.previous_js, ord=1) > save_threshold:
#         print('save:[{}]: {}'.format(self.frameNo, js))
#         d = {'frameNo':self.frameNo, 'jointPosition':js, 'image':img}
#         for k,id in self.objects.items():
#             d[k] = p.getBasePositionAndOrientation(id)
#         self.frames.append(d)
#         self.frameNo += 1
#         self.previous_js = js
        
# def process_image(msg):
#     global frameNo
#     try:
#         cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
#         # img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow('image', img)
#         # cv2.waitKey(1)
#     except CvBridgeError as err:
#         print(err)
#     else:
#         cv2.imwrite('data/0/image{:05d}.jpg'.format(frameNo), cv2_img)
#         frameNo += 1

joy_msg = Joy()
def joy_callback(msg):
    global joy_msg
    joy_msg = msg
    

rospy.init_node('recorder')
rospy.loginfo('recorder node started')
image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
joint_state_sub = message_filters.Subscriber("/joint_states", JointState)
trajectory_sub = message_filters.Subscriber("/scaled_pos_joint_traj_controller/command", JointTrajectory)
#ts = message_filters.TimeSynchronizer([image_sub, joint_state_sub, trajectory_sub], queue_size=10)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_state_sub, trajectory_sub], queue_size=1, slop=0.05)
ts.registerCallback(state_callback)

rospy.Subscriber("/spacenav/joy", Joy, joy_callback, queue_size=1)

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

pub = rospy.Publisher('/jog_frame', JogFrame, queue_size=1)

# initial position for 'red cube reacing task'
# initial_position = np.array([-0.0778662, 0.36922, -0.187572])

# initial position for 'kitting task'
initial_position = np.array([-0.097, 0.5, -0.147])
initial_rotation = np.array([-0.720, 0.012, -0.011, 0.694])
moving_to_initial_position = False

rate = rospy.Rate(15.0)

while not rospy.is_shutdown():
    trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
    # p = trans.transform.translation
    # print('translation = ', np.array([p.x, p.y, p.z]))
    # p = trans.transform.rotation
    # print('rotation = ', np.array([p.x, p.y, p.z, p.w]))

    if moving_to_initial_position:
        try:
            trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
            p = trans.transform.translation
            p = np.array([p.x, p.y, p.z])
            print(p)
            next_subgoal = initial_position
            dp = next_subgoal - p
            if np.linalg.norm(dp) < 1e-2: # reached goal
                print('reached the initial position')
                moving_to_initial_position = False
                continue
            v = 0.003 * dp / np.linalg.norm(dp)
            msg = JogFrame()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            msg.group_name = 'manipulator'
            msg.link_name = 'tool0'
            msg.angular_delta.x = 0
            msg.angular_delta.y = 0
            msg.angular_delta.z = 0
            msg.linear_delta.x = v[0]
            msg.linear_delta.y = v[1]
            msg.linear_delta.z = v[2]
            pub.publish(msg)
        except:
            rate.sleep()
        continue

    else:
        try:
            if joy_msg.buttons[1] == 1: # right button is down
                if recording_now:
                    stop_recording()
                print('moving to the initial position')
                moving_to_initial_position = True
            elif joy_msg.buttons[0] == 1: # left button is down
                if not recording_now:
                    start_recording()
        except:
            pass
        rate.sleep()
        
    
# initial pose
# name: 
#   - elbow_joint
#   - shoulder_lift_joint
#   - shoulder_pan_joint
#   - wrist_1_joint
#   - wrist_2_joint
#   - wrist_3_joint
# position: [2.2289021650897425, -0.26636190832171636, 1.3831090927124023, -1.857983251611227, 1.4491826295852661, 3.1288983821868896]
#
# base_link -> tool0
# pose:
#   position:
#     x: -0.0778662
#     y: 0.36922
#     z: -0.197572
#   orientation:
#     x: -0.74241
#     y: 0.0243073
#     z: -0.0218602
#     w: 0.669148

    
    
# if __name__ == '__main__':
#     try:
#         start_node()
#     except rospy.ROSInterruptException:
#         pass


# def save_image(image, frameNo):
#     pass

# def save_joint_state(joint_state, frameNo):
#     pass

# def record_episode(episode_number=1):
#     frameNo = 0
#     while not episode_end:
#         image = get_latest_image()
#         js = get_latest_joint_state()
#         save_image(image, frameNo)
#         save_joint_state(js, frameNo)
#         frameNo += 1
        
