# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()
frameNo = 0

def process_image(msg):
    global frameNo
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
    except CvBridgeError as err:
        print(err)
    else:
        cv2.imwrite('data/0/image{:05d}.jpg'.format(frameNo), cv2_img)
        frameNo += 1

def start_node():
    rospy.init_node('recorder')
    rospy.loginfo('recorder node started')
    rospy.Subscriber("/camera/color/image_raw", Image, process_image)
    rospy.spin()

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
        
