import glob
import os
import subprocess
import sys
from pathlib import Path

import cameramodels
from mask2polygon import *

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
    polygon_to_mask_all(polygon_files_dir=str(source), output_dir=str(source))


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


#####
##### Image-based evaluation
#####


# def compute_3d_position(mask, width=640, height=480):
#     xs = np.arange(width) * mask
#     x = np.average(xs[xs != 0])
#     ys = np.arange(height).reshape((height, 1)) * mask
#     y = np.average(ys[ys != 0])
#     return x, y


def compute_3d_position(mask, points):
    msk = mask[:, :, np.newaxis]
    masked_points = points * msk
    n_valid_points = np.count_nonzero(masked_points[:, :, 2])
    return [np.sum(masked_points[:, :, i]) / n_valid_points for i in range(3)]


def image_based_tracking(project_dir="~/Program/yolov5/runs/predict-seg/20240131_193129"):
    p = Path(project_dir)
    p = p.expanduser()
    trajectories = {}
    poly_files = sorted(glob.glob(str(p / "labels" / "*-color.txt")))
    for frame_number, poly_file in enumerate(poly_files):
        n = int(Path(poly_file).stem.replace("-color", ""))
        clss, masks, bbs = polygon_to_mask(poly_file, unify_mask=False)
        depth_path = f"/home/ryo/Dataset/forcemap_evaluation/20240131_193129/{n:06d}-depth.png"
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        points = back_project(depth_img)
        for cls, mask, bb in zip(clss, masks, bbs):
            pos = compute_3d_position(mask, points)
            label = label_names[cls]
            # print(label, pos)
            try:
                traj = trajectories[label]
            except KeyError:
                traj = {}
                trajectories[label] = traj
            trajectories[label][frame_number] = pos

    return trajectories


def back_project(depth_img):
    intrinsic_matrix = meta_data["intrinsic_matrix"]
    depth_scale = meta_data["factor_depth"]
    width, height = meta_data["image_size"]
    cm = cameramodels.PinholeCameraModel.from_intrinsic_matrix(intrinsic_matrix, height, width)

    uv = np.hstack([np.tile(np.arange(width), height)[:, None], np.repeat(np.arange(height), width)[:, None]])
    depth = np.array(depth_img / depth_scale, "f")
    points = cm.batch_project_pixel_to_3d_ray(uv) * depth.reshape(-1, 1)
    points = points.reshape(height, width, 3)
    return points


def plot_trajectory(trajs, object_name, z_offset=-0.8):
    traj = trajs[object_name]
    indices = np.array(list(traj.keys()))
    xs, ys, zs = zip(*np.array(list(traj.values())))
    ts = indices * 0.033  # convert index to second
    fig, ax = plt.subplots()
    ax.plot(ts, xs, "o-", label="x")
    ax.plot(ts, ys, "o-", label="y")
    ax.plot(ts, np.array(zs) + z_offset, "o-", label=f"z{z_offset}")
    ax.set_title(object_name)
    ax.set_xlabel("[sec]")
    ax.set_ylabel("[m]")
    ax.legend(loc="upper right")
    plt.show()


# ffmpeg -r 30 -i %06d-color.jpg -vcodec libx264 -pix_fmt yuv420p out.mp4
