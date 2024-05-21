import argparse
import copy
import os
import sys
from pathlib import Path

import cv2  # state of the art computer vision algorithms library
import matplotlib.pyplot as plt  # 2D plotting library producing publication quality figures
import numpy as np  # fundamental package for scientific computing
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API

parser = argparse.ArgumentParser()
parser.add_argument("--bag_file", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--start_index", type=int, default=1)
parser.add_argument("--end_index", type=int, default=1000)
parser.add_argument("--colorize", action="store_true")
args = parser.parse_args()


output_root_dir = Path(args.output_dir)
start_index = args.start_index
end_index = args.end_index

output_dir = output_root_dir / Path(args.bag_file).stem
if not output_dir.exists():
    output_dir.mkdir()


def bag_to_images(bag_file, colorize_depth=False):
    time_stamps = []

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(str(Path(args.bag_file).expanduser()), repeat_playback=False)
    # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipe.start(cfg)

    for i in range(start_index):
        pipe.wait_for_frames()

    for i in range(start_index, end_index):
        flag, frameset = pipe.try_wait_for_frames()
        if flag == False:
            break

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        color_frame = frameset.get_color_frame()
        color = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        aligned_depth_frame = frameset.get_depth_frame()

        if colorize_depth:
            colorizer = rs.colorizer()
            aligned_depth_frame = colorizer.colorize(aligned_depth_frame)

        aligned_depth = np.asanyarray(aligned_depth_frame.get_data())

        print(i, color_frame.timestamp)
        # data.append(((color_frame.timestamp, color), (aligned_depth_frame.timestamp, aligned_depth)))

        cv2.imwrite(str(output_dir / f"{i-start_index:06d}-color.jpg"), color)
        cv2.imwrite(str(output_dir / f"{i-start_index:06d}-depth.png"), aligned_depth)

        ts = color_frame.timestamp / 1e3  # msec -> sec
        time_stamps.append(ts)

        # Show the two frames together:
        # images = np.hstack((color, colorized_depth))
        # plt.imshow(images)

    # Cleanup:
    pipe.stop()

    time_stamps = np.array(time_stamps)
    time_stamps = time_stamps - time_stamps[0]
    np.save(output_dir / "time_stamps.npy", time_stamps)


if __name__ == "__main__":
    bag_to_images(args.bag_file, args.colorize)
