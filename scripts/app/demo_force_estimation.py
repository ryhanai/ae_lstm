#!/usr/bin/env python3

import copy

import cv2
from force_estimation import force_distribution_viewer
import force_estimation.force_estimation_v4 as fe
from force_estimation import forcemap
import numpy as np
import pandas as pd
import rospy
import scipy

# import force_estimation_v2 as fe
import torch
from app.pick_planning import LiftingDirectionPlanner
from cv_bridge import CvBridge, CvBridgeError
# from dynamic_reconfigure.server import Server
from force_estimation.eipl_utils import normalization, tensor2numpy
# from force_estimation.cfg import force_estimationConfig
from scipy.optimize import minimize
from sensor_msgs.msg import Image

image_topic = "/camera/color/image_raw"


model = fe.ForceEstimationResNetTabletop()
# ckpt = torch.load('../runs/20230627_1730_52/CAE.pth')  # stable, moonshot intermediate report demo
# ckpt = torch.load("../runs/20230919_2116_44/CAE.pth")  # stable, AIST public event demo
ckpt = torch.load("../force_estimation/log/20240304_1834_24/ForceEstimationResNetTabletop.pth")  # stable, AIST public event demo
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


fmap = forcemap.GridForceMap("small_table")
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
# params = copy.copy(force_estimationConfig.defaults)
planner = LiftingDirectionPlanner(fmap)

bridge = CvBridge()


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. ind    ataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indat    aRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


def crop_center_d415(img):
    c = (-40, 75)
    crop = 64
    return img[180 + c[0] : 540 + c[0], 320 + c[1] + crop : 960 + c[1] - crop]


def crop_center_sr305(img):
    c = (-40, 75)
    crop = 64
    img = cv2.resize(img, (1280, 720))
    return img[180 + c[0] : 540 + c[0], 320 + c[1] + crop : 960 + c[1] - crop]



frame_no = 0


def process_image(msg, save_result=False):
    global frame_no

    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # sensor = params["sensor_type"]
    sensor = 0
    if sensor == 0:
        img = crop_center_d415(cv_image)
    elif sensor == 1:
        img = crop_center_sr305(cv_image)
    else:
        print("unknown sensor: ", sensor)
        cv2.waitKey(1)
        return

    # Tensorflow
    # fimg = img.astype(np.float64) / 255.
    # xs = np.expand_dims(fimg, 0)
    # predicted_force_map = model.predict(xs)

    # PyTorch
    fimg = img.transpose(2, 0, 1)
    fimg = normalization(fimg.astype(np.float32), (0.0, 255.0), [0.1, 0.9])
    fimg = torch.from_numpy(fimg).float()
    batch = torch.unsqueeze(fimg, 0)
    batch.to("cuda")
    yi = model(batch)[0]
    batch.to("cpu")
    fv = tensor2numpy(yi).transpose(1, 2, 0)
    predicted_force_map = fv[:, :, :20]

    if save_result:
        cv2.imwrite(f"data/rgb{frame_no:05}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        pd.to_pickle(predicted_force_map, f"data/forcemap{frame_no:05}.pkl")
        frame_no += 1

    # fv = np.zeros((40, 40, 40))
    # fv[:, :, :20] = predicted_force_map[0]

    fmap.set_values(fv)

    viewer.publish_bin_state(
        # None, fmap, draw_fmap=True, draw_force_gradient=False, draw_range=[params["force_vis_threshold"], 0.9]
        None, fmap, draw_fmap=True, draw_force_gradient=False, draw_range=[0.43, 0.9]        
    )

    # if params["calc_lifting_direction"] == True:
    if params["calc_lifting_direction"] == True:        
        # unnormalize the predicted force
        bounds = np.log([1e-8, 1e-3])
        predicted_force_map = 1e6 * np.exp((predicted_force_map - 0.1) / 0.8 * (bounds[1] - bounds[0]) + bounds[0])
        print(f"AVE predicted force: {np.average(predicted_force_map)}")

        object_center = viewer.rviz_client.getObjectPosition()
        v, omega = pick_direction_plan(predicted_force_map, object_center, object_radius=0.05)
        print(v, omega)
        viewer.rviz_client.show()

    bgr_center_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Input Image", bgr_center_image)
    cv2.waitKey(1)


def parameter_callback(config, level):
    global params
    rospy.loginfo(
        """Reconfigure Request: {calc_lifting_direction}, {force_vis_threshold}, {sensor_type}""".format(**config)
    )
    params = config
    return config


def start_node():
    rospy.Subscriber(image_topic, Image, process_image)
    # param_srv = Server(force_estimationConfig, parameter_callback)
    rospy.spin()


if __name__ == "__main__":
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
