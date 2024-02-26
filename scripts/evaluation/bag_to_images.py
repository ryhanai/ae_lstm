import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rosbag
from cv_bridge import CvBridge

parser = argparse.ArgumentParser()
parser.add_argument("--bag_file", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--start_index", type=int, default=1)
parser.add_argument("--end_index", type=int, default=1000)
args = parser.parse_args()


output_root_dir = Path(args.output_dir)
bag = rosbag.Bag(args.bag_file)
start_index = args.start_index
end_index = args.end_index

output_dir = output_root_dir / Path(args.bag_file).stem
if not output_dir.exists():
    output_dir.mkdir()

bridge = CvBridge()


def add_frame(data, color_tm, color_img, depth_tm, depth_img):
    if color_tm == None:
        return
    if depth_tm == None:
        return
    if np.abs(color_tm - depth_tm) < 1:  # within 1ms
        data.append(((color_tm, color_img), (depth_tm, depth_img)))


def bag_to_images(bag):
    data = []
    color_tm = None
    depth_tm = None
    color_img = None
    depth_img = None

    for topic, msg, t in bag.read_messages():
        if topic == "/device_0/sensor_1/Color_0/image/data":
            color_tm = msg.header.stamp.to_nsec() / 1e6
            color_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            add_frame(data, color_tm, color_img, depth_tm, depth_img)
        if topic == "/device_0/sensor_0/Depth_0/image/data":
            depth_tm = msg.header.stamp.to_nsec() / 1e6
            depth_img = bridge.imgmsg_to_cv2(msg, "passthrough")
            add_frame(data, color_tm, color_img, depth_tm, depth_img)

    return data


def write_data(data):
    global end_index
    time_stamps = []
    end_index = min(end_index, len(data))
    for i, d in enumerate(data[start_index:end_index]):
        (color_tm, color_img), (depth_tm, depth_img) = d
        cv2.imwrite(str(output_dir / f"{i:06d}-color.jpg"), color_img)
        cv2.imwrite(str(output_dir / f"{i:06d}-depth.png"), depth_img)
        time_stamps.append((color_tm + depth_tm) / 2.0 / 1e3)  # msec -> sec

    time_stamps = np.array(time_stamps)
    time_stamps = time_stamps - time_stamps[0]
    np.save(output_dir / "time_stamps.npy", time_stamps)


def crop_center_d415(img):
    c = (-40, 75)
    crop = 64
    return img[180 + c[0] : 540 + c[0], 320 + c[1] + crop : 960 + c[1] - crop]


def extract_rgb(bag, msg_seqs=[]):
    rgbs = []
    for topic, msg, t in bag.read_messages():
        if msg.header.seq in msg_seqs:
            if topic == "/camera/color/image_raw":
                rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
                rgb = crop_center_d415(rgb)
                rgbs.append(rgb)

    for i, rgb in enumerate(rgbs):
        cv2.imwrite(str(output_dir / f"{i:06d}-color.jpg"), rgb)
    return rgbs


seqs = [27060, 27232, 27310, 27406, 27543, 27688, 28010, 28290, 28427, 28856, 29051, 29288, 29506, 29700, 29847]
print(len(extract_rgb(bag, seqs)))

# if __name__ == "__main__":
#     data = bag_to_images(bag)
#     write_data(data)
