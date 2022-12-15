#!/usr/bin/env python

import numpy as np
import forcemap
import force_estimation_v2 as fe
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import force_distribution_viewer


image_topic = '/camera/color/image_raw'


model = fe.model_rgb_to_fmap_res50()
model.load_weights('../runs/ae_cp.basket-filling2.model_resnet.20221202165608/cp.ckpt')
fmap = forcemap.GridForceMap('seria_basket')
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


bridge = CvBridge()


def crop_center(img):
    c = (-40, 75)
    crop = 64
    return img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]

import pandas as pd
def process_image(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    except CvBridgeError as e:
        rospy.logerr('CvBridge Error: {0}'.format(e))

    img = crop_center(cv_image)
    
    fimg = img.astype(np.float64) / 255.
    
    xs = np.expand_dims(fimg, 0)
    predicted_force_map = model.predict(xs)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = predicted_force_map[0]
    fmap.set_values(fv)
    viewer.publish_bin_state(None, fmap, draw_fmap=True, draw_force_gradient=False)
    # fmap.visualize()
    # plt.show()

    bgr_center_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Input Image', bgr_center_image)
    cv2.waitKey(1)


def start_node():
    rospy.Subscriber(image_topic, Image, process_image)
    rospy.spin()


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
