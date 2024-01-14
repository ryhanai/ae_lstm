import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import supervision as sv

data_dir = Path(os.environ["HOME"]) / "Dataset/ycb_video/data"


def read_meta_file(sequence_number, frame_number):
    p = data_dir / f"{sequence_number:04d}/{frame_number:06d}-meta.mat"
    return scipy.io.loadmat(p)


def get_cls_indices(sequence_number, frame_number):
    meta_data = read_meta_file(sequence_number, frame_number)
    cls_indices = meta_data["cls_indexes"]
    return cls_indices


def read_mask_file(sequence_number, frame_number):
    p = data_dir / f"{sequence_number:04d}/{frame_number:06d}-label.png"
    return cv2.imread(str(p))[:, :, 0]


def read_color_image(sequence_number, frame_number):
    p = data_dir / f"{sequence_number:04d}/{frame_number:06d}-color.jpg"
    return cv2.imread(str(p))


def draw_polygons(bgr_img, polygons, color=[0, 255, 0]):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    for p in polygons:
        annotated_img = sv.draw_polygon(rgb_img, p, sv.draw.color.Color(*color))
    plt.imshow(annotated_img)
    plt.show()


def save_polygons(f, cls_index, polygons, mask):
    height, width = mask.shape
    f.write(f"{cls_index[0]} ")
    for polygon in polygons:
        polygon = np.array(polygon) / [width, height]
        for v in polygon:
            f.write(f"{v[0]:0.6f} {v[1]:0.6f} ")
    f.write("\n")


def convert(sequence_number, frame_number, draw_result=False):
    cls_indices = get_cls_indices(sequence_number, frame_number)
    mask = read_mask_file(sequence_number, frame_number)

    # save_path = data_dir / f"{sequence_number:04d}/{frame_number:06d}-poly.txt"
    save_path = f"labels/{sequence_number:04d}-{frame_number:06d}-poly.txt"
    with open(save_path, "w") as f:
        for cls_index in cls_indices:
            vertices = []
            single_mask = np.where(mask == cls_index, 1, 0)
            polygons = sv.mask_to_polygons(single_mask)

            if draw_result:
                bgr_img = read_color_image(sequence_number, frame_number)
                draw_polygons(bgr_img, polygons)

            save_polygons(f, cls_index, polygons, mask)
