import glob
import os
import shutil
from pathlib import Path

from mask2polygon import gen_polygon_mask

YCB_VIDEO_DIR = Path(os.environ["HOME"]) / "Dataset" / "ycb_video"
YOLO_DATA_DIR = YCB_VIDEO_DIR / "yolo_seg"


def do_create():
    start_idx = 0
    to_img_dir = YOLO_DATA_DIR / "images" / "train"
    to_label_dir = YOLO_DATA_DIR / "labels" / "train"
    for seq_num in range(80):
        from_dir = YCB_VIDEO_DIR / "data" / f"{seq_num:04d}"
        image_files = glob.glob(str(from_dir / "*-color.jpg"))
        for i, image_file in enumerate(image_files):
            shutil.copy(image_file, f"{to_img_dir}/{start_idx + i:012d}.jpg")
            print(f"copy: {image_file}, {to_img_dir}/{start_idx + i:012d}.jpg")
            gen_polygon_mask(image_file, f"{to_label_dir}/{start_idx + i:012d}.txt")
        start_idx += len(image_files)

    start_idx = 123705
    from_dir = YCB_VIDEO_DIR / "data_syn"
    image_files = glob.glob(str(from_dir / "*-color.jpg"))
    for i, image_file in enumerate(image_files):
        shutil.copy(image_file, f"{to_img_dir}/{start_idx + i:012d}.jpg")
        print(f"copy: {image_file}, {to_img_dir}/{start_idx + i:012d}.jpg")
        gen_polygon_mask(image_file, f"{to_label_dir}/{start_idx + i:012d}.txt")
    start_idx += len(image_files)

    start_idx = 0
    to_img_dir = YOLO_DATA_DIR / "images" / "test"
    to_label_dir = YOLO_DATA_DIR / "labels" / "test"
    for seq_num in range(80, 92):
        from_dir = YCB_VIDEO_DIR / "data" / f"{seq_num:04d}"
        image_files = glob.glob(str(from_dir / "*-color.jpg"))
        for i, image_file in enumerate(image_files):
            shutil.copy(image_file, f"{to_img_dir}/{start_idx + i:012d}.jpg")
            print(f"copy: {image_file}, {to_img_dir}/{start_idx + i:012d}.jpg")
            gen_polygon_mask(image_file, f"{to_label_dir}/{start_idx + i:012d}.txt")
        start_idx += len(image_files)


def view_converted_polygons():
    # start_idx = 0
    # to_img_dir = YOLO_DATA_DIR / "images" / "train"
    # for seq_num in range(80):
    #     from_dir = YCB_VIDEO_DIR / "data" / f"{seq_num:04d}"
    #     image_files = glob.glob(str(from_dir / "*-color.jpg"))
    #     for i, image_file in enumerate(image_files):
    #         print(f"{image_file}")
    #         gen_polygon_mask(image_file, None, save_result=False, view_result=True)
    #     start_idx += len(image_files)

    from_dir = YCB_VIDEO_DIR / "data_syn"
    image_files = glob.glob(str(from_dir / "*-color.jpg"))
    for i, image_file in enumerate(image_files):
        print(f"{image_file}")
        gen_polygon_mask(image_file, None, save_result=False, view_result=True)
    start_idx += len(image_files)

    # start_idx = 0
    # to_img_dir = YOLO_DATA_DIR / "images" / "test"
    # to_label_dir = YOLO_DATA_DIR / "labels" / "test"
    # for seq_num in range(80, 92):
    #     from_dir = YCB_VIDEO_DIR / "data" / f"{seq_num:04d}"
    #     image_files = glob.glob(str(from_dir / "*-color.jpg"))
    #     for i, image_file in enumerate(image_files):
    #         shutil.copy(image_file, f"{to_img_dir}/{start_idx + i:012d}.jpg")
    #         print(f"copy: {image_file}, {to_img_dir}/{start_idx + i:012d}.jpg")
    #         gen_polygon_mask(image_file, f"{to_label_dir}/{start_idx + i:012d}.txt")
    #     start_idx += len(image_files)
