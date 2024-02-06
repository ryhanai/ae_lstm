import glob
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


def gen_bb_and_mask(project="~/Dataset/forcemap_evaluation", name="20240131_191917", conf_thres=0.1):
    # use YOLO
    # anaconda, densefusion env
    # go to Program/yolov5
    predict = "~/Program/yolov5/segment/predict.py"
    weights = "~/Program/moonshot/ae_lstm/scripts/evaluation/yolo_runs/exp8_12epoch/weights/best.pt"
    source = f"{project}/{name}"
    # cmd = f"python {predict} --weights {weights} --source {source} --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok --nosave"
    cmd = f"python {predict} --weights {weights} --source {source} --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("failed to execute YOLO", file=sys.stderr)
        return

    # generate (masks, bbs, meta-data) from polygons
    source = str(Path(source).expanduser())
    mask2polygon.polygon_to_mask_all(polygon_files_dir=source, output_dir=source)


def estimate_pose():
    # use DenseFusion
    # anaconda, densefusion env
    cmd = f"densefusion"
    subprocess.run(cmd, shell=True)


def gen_densefusion_config_file(
    data_dir,
    densefusion_config_dir="~/anaconda3/envs/densefusion/lib/python3.8/site-packages/dense_fusion/datasets/ycb/config",
):
    imgs = glob.glob(str(Path(data_dir).expanduser() / "*.jpg"))
    print(imgs)
    # max_index = max([int(Path(p).stem.replace('-color', '')) for p in imgs])

    data_list_file = Path(densefusion_config_dir) / "test_data_list.txt"
    with open(data_list_file.expanduser(), "w") as f:
        for name in [Path(img).stem.replace("-color", "") for img in imgs]:
            f.write(f"{name}\n")


def do_track():
    bag_to_images()
    gen_bb_and_mask()
    estimate_pose()
