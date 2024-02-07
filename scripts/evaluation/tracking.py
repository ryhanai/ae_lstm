import glob
import os
import subprocess
import sys
from pathlib import Path

import mask2polygon

bag_file = "~/Documents/20240131_191917.bag"
output_dir = "~/Dataset/forcemap_evaluation"
start_index = 160
end_index = 300


def bag_to_images():
    # No anaconda env
    cmd = f"python bag_to_images.py --bag_file {bag_file} --output_dir {output_dir} --start_index {start_index} --end_index {end_index}"
    subprocess.run(cmd, shell=True)


def gen_bb_and_mask(project="~/Dataset/forcemap_evaluation", name="20240131_191917", conf_thres=0.15):
    # use YOLO
    # anaconda, densefusion env
    # go to Program/yolov5
    predict = "~/Program/yolov5/segment/predict.py"
    weights = "~/Program/moonshot/ae_lstm/scripts/evaluation/yolo_runs/exp8_12epoch/weights/best.pt"
    source = Path(f"{project}/{name}").expanduser()

    source_pattern = str(source / "*-color.jpg")
    # cmd = f"python {predict} --weights {weights} --source {source} --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok --nosave"
    cmd = f"python {predict} --weights {weights} --source '{source_pattern}' --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok --nosave"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("failed to execute YOLO", file=sys.stderr)
        return

    # generate (masks, bbs, meta-data) from polygons
    mask2polygon.polygon_to_mask_all(polygon_files_dir=str(source), output_dir=str(source))


def estimate_pose(project="~/Dataset/forcemap_evaluation", name="20240131_191917"):
    # use DenseFusion
    # anaconda, densefusion env
    proj_dir = str(Path(f"{project}/{name}").expanduser())
    print(proj_dir)

    print("generate config file for densefusion: {}")
    gen_densefusion_config_file(proj_dir)

    os.environ["YCB_VIDEO_DATASET_PATH"] = proj_dir
    predict = "~/Program/dense-fusion/examples/eval_ycb.py"
    cmd = f"python {predict}"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("failed to execute DenseFusion", file=sys.stderr)
        return


def gen_densefusion_config_file(
    data_dir,
    densefusion_config_dir="~/anaconda3/envs/densefusion/lib/python3.8/site-packages/dense_fusion/datasets/ycb/config",
):
    imgs = glob.glob(str(Path(data_dir).expanduser() / "*.jpg"))
    list.sort(imgs)
    print(imgs)
    # max_index = max([int(Path(p).stem.replace('-color', '')) for p in imgs])

    data_list_file = Path(densefusion_config_dir) / "test_data_list.txt"
    with open(data_list_file.expanduser(), "w") as f:
        iter_name = iter([Path(img).stem.replace("-color", "") for img in imgs])
        try:
            name = next(iter_name)
            f.write(f"{name}")
            while True:
                name = next(iter_name)
                f.write(f"\n{name}")
        except StopIteration:
            pass


def do_track(name):
    # bag_to_images()
    gen_bb_and_mask(name=name)
    estimate_pose(name=name)


# ffmpeg -r 30 -i %06d-color.jpg -vcodec libx264 -pix_fmt yuv420p out.mp4
