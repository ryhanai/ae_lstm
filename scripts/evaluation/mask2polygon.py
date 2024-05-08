import glob
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


def polygon_to_mask(polygon_file_path, unify_mask=True, width=640, height=480):
    def parse_poly(poly_line):
        vals = poly_line.split()
        cls = int(vals[0]) + 1
        confidence = float(vals[-1])
        vs = [float(val) for val in vals[1:-1]]
        xs = (np.array(vs[::2]) * width).astype("int32")
        ys = (np.array(vs[1::2]) * height).astype("int32")
        poly = np.array(list(zip(xs, ys)))
        return cls, poly, confidence

    with open(polygon_file_path, "r") as f:
        poly_strs = f.readlines()

    polys = [parse_poly(poly) for poly in poly_strs]
    clss = [cls for cls, _, _ in polys]
    masks = [sv.polygon_to_mask(vs, resolution_wh=[width, height]).astype("uint8") for _, vs, _ in polys]
    bbs = [sv.polygon_to_xyxy(vs) for _, vs, _ in polys]

    if unify_mask:
        unified_mask = np.zeros((height, width), dtype="uint8")
        for i in range(len(clss)):
            unified_mask += clss[i] * masks[i]
        unified_mask = np.where(np.isin(unified_mask, clss), unified_mask, 0)
        return clss, unified_mask, bbs
    else:
        return clss, masks, bbs


# definitions in YOLO
label_names = (
    "__background__",
    "002_master_chef_can",  # 0
    "003_cracker_box",  # 1
    "004_sugar_box",  # 2
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
)

# definitions in DenseFusion
df_label_names = {
    "002_master_chef_can": 1,
    "003_cracker_box": 2,
    "004_sugar_box": 3,
    "005_tomato_soup_can": 4,
    "006_mustard_bottle": 5,
    "007_tuna_fish_can": 6,
    "008_pudding_box": 7,
    "009_gelatin_box": 8,
    "010_potted_meat_can": 9,
    "011_banana": 10,
    "019_pitcher_base": 11,
    "021_bleach_cleanser": 12,
    "024_bowl": 13,
    "025_mug": 14,
    "035_power_drill": 15,
    "036_wood_block": 16,
    "037_scissors": 17,
    "040_large_marker": 18,
    "051_large_clamp": 19,
    "052_extra_large_clamp": 20,
    "061_foam_brick": 21,
}


def cls_yolo2df(cls):
    return df_label_names[label_names[cls + 1]]


meta_data = {
    "cls_indexes": None,
    "factor_depth": np.array([[1000]], dtype=np.uint16),  # realsense
    # "intrinsic_matrix": np.array(
    #     [
    #         [1.066778e03, 0.000000e00, 3.129869e02],
    #         [0.000000e00, 1.067487e03, 2.413109e02],
    #         [0.000000e00, 0.000000e00, 1.000000e00],
    #     ]
    # ),  # YCB video original (Xtion?)
    # "intrinsic_matrix": np.array(
    #     [
    #         [608.5486450195312, 0.0, 325.8691101074219],
    #         [0.0, 608.2636108398438, 232.5676727294922],
    #         [0.000000e00, 0.000000e00, 1.000000e00],
    #     ]
    # ),  # 640x480
    "intrinsic_matrix": np.array(
        [
            [597.7420654296875, 0.0, 322.5055236816406],
            [0.0, 608.2636108398438, 232.5676727294922],
            [0.000000e00, 0.000000e00, 1.000000e00],
        ]
    ),  # 640x480
    "image_size": np.array([640, 480]),
    # "rotation_translation_matrix": np.array(
    #     [
    #         [0.998481, -0.0421787, 0.0354595, 0.05612157],
    #         [0.00194755, -0.61609, -0.787673, 0.02851242],
    #         [0.0550693, 0.786545, -0.615072, 0.60312363],
    #     ]
    # ),
    # "vertmap": np.array([]),
    # "poses": np.array([]),
    # "center": np.array([[257.84763275, 226.84390407], [414.23978618, 204.16392412], [463.22414406, 317.07572823]]),
}


# realsense D415
# color
# height: 480
# width: 640
# distortion_model: "plumb_bob"
# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [608.5486450195312, 0.0, 325.8691101074219, 0.0, 608.2636108398438, 232.5676727294922, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [608.5486450195312, 0.0, 325.8691101074219, 0.0, 0.0, 608.2636108398438, 232.5676727294922, 0.0, 0.0, 0.0, 1.0, 0.0]

# depth
# height: 480
# width: 640
# distortion_model: "plumb_bob"
# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [597.7420654296875, 0.0, 322.5055236816406, 0.0, 597.7420654296875, 242.5670623779297, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [597.7420654296875, 0.0, 322.5055236816406, 0.0, 0.0, 597.7420654296875, 242.5670623779297, 0.0, 0.0, 0.0, 1.0, 0.0]


def polygon_to_mask_all(polygon_files_dir, output_dir=None, unify_mask=False):
    p = Path(polygon_files_dir)
    poly_files = glob.glob(str(p / "labels" / "*-color.txt"))
    if output_dir == None:
        output_dir = p
    else:
        output_dir = Path(output_dir)

    for poly_file in poly_files:
        print(poly_file)
        if unify_mask:
            clss, mask, bbs = polygon_to_mask(poly_file, unify_mask=True)
            p = Path(poly_file)
            label_path = output_dir / p.stem.replace("-color", "-label.png")
            cv2.imwrite(str(label_path), mask)
        else:
            clss, masks, bbs = polygon_to_mask(poly_file, unify_mask=False)
            for cls, msk, bb in zip(clss, masks, bbs):
                object_name = label_names[cls]
                p = Path(poly_file)
                label_path = output_dir / p.stem.replace("-color", f"-label-{object_name}.png")
                cv2.imwrite(str(label_path), 255 * msk)

        bbox_path = output_dir / p.stem.replace("-color", "-box.txt")
        with open(str(bbox_path), "w") as f:
            for cls, bb in zip(clss, bbs):
                f.write(f"{label_names[cls]} {bb[0]:0.2f} {bb[1]:0.2f} {bb[2]:0.2f} {bb[3]:0.2f}\n")

        cls_indexes = np.array(clss)
        meta_data["cls_indexes"] = cls_indexes.reshape((len(clss), 1))
        meta_path = output_dir / p.stem.replace("-color", "-meta.mat")
        scipy.io.savemat(str(meta_path), meta_data)
