import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import supervision as sv

# data_dir = Path(os.environ["HOME"]) / "Dataset/ycb_video/data"


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


def annotate_polygons(rgb_img, polygons, color=[0, 255, 0]):
    annotated_img = rgb_img
    for p in polygons:
        annotated_img = sv.draw_polygon(annotated_img, p, sv.draw.color.Color(*color))
    return annotated_img


def save_polygons(f, cls_index, polygons, mask):
    height, width = mask.shape
    if len(polygons) > 0:
        f.write(f"{cls_index-1} ")
        for polygon in polygons:
            polygon = np.array(polygon) / [width, height]
            for v in polygon:
                f.write(f"{v[0]:0.6f} {v[1]:0.6f} ")
        f.write("\n")
    else:
        print("skip saving undetected class")


# def convert(sequence_number, frame_number, draw_result=False):
#     cls_indices = get_cls_indices(sequence_number, frame_number)
#     mask = read_mask_file(sequence_number, frame_number)

#     # save_path = data_dir / f"{sequence_number:04d}/{frame_number:06d}-poly.txt"
#     save_path = f"labels/{sequence_number:04d}-{frame_number:06d}-poly.txt"
#     with open(save_path, "w") as f:
#         for cls_index in cls_indices:
#             single_mask = np.where(mask == cls_index, 1, 0)
#             polygons = sv.mask_to_polygons(single_mask)

#             if draw_result:
#                 bgr_img = read_color_image(sequence_number, frame_number)
#                 draw_polygons(bgr_img, polygons)

#             save_polygons(f, cls_index, polygons, mask)


def gen_polygon_mask(color_file_path, polygon_file_path, save_result=True, view_result=False):
    meta_data = scipy.io.loadmat(str.replace(color_file_path, "-color.jpg", "") + "-meta.mat")
    cls_indices = meta_data["cls_indexes"]
    cls_indices = cls_indices.flatten().astype(np.uint8)
    mask = cv2.imread(str.replace(color_file_path, "-color.jpg", "") + "-label.png")[:, :, 0]

    if save_result:
        with open(polygon_file_path, "w") as f:
            for cls_index in cls_indices:
                single_mask = np.where(mask == cls_index, 1, 0)
                polygons = sv.mask_to_polygons(single_mask)
                save_polygons(f, cls_index, polygons, mask)

    if view_result:
        bgr_img = cv2.imread(color_file_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        for cls_index in cls_indices:
            single_mask = np.where(mask == cls_index, 1, 0)
            polygons = sv.mask_to_polygons(single_mask)
            rgb_img = annotate_polygons(rgb_img, polygons)
        plt.imshow(rgb_img)
        plt.show()
