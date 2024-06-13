import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import cameramodels
import quaternion
from mask2polygon import *
from sklearn import mixture
from sklearn.cluster import DBSCAN

# bag_file = "~/Documents/20240131_191917.bag"
# output_dir = "~/Dataset/forcemap_evaluation"
# start_index = 160
# end_index = 300


def unzip(x):
    return zip(*x)


def gen_bb_and_mask_with_YOLO(project="~/Dataset/forcemap_evaluation/01_GAFS_1", conf_thres=0.15):
    # use YOLO
    # anaconda, densefusion env
    # go to Program/yolov5
    predict = "~/Program/yolov5/segment/predict.py"
    weights = "~/Program/moonshot/ae_lstm/scripts/evaluation/yolo_runs/exp8_12epoch/weights/best.pt"
    source = Path(project).expanduser()
    source_pattern = str(source / "*-color.jpg")

    name = "yolo_out"
    # cmd = f"python {predict} --weights {weights} --source {source} --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok --nosave"
    cmd = f"python {predict} --weights {weights} --source '{source_pattern}' --conf-thres {conf_thres} --save-txt --save-conf --project {project} --name {name} --exist-ok"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("failed to execute YOLO", file=sys.stderr)
        return

    # generate (masks, bbs, meta-data) from polygons
    polygon_to_mask_all(polygon_files_dir=str(source / name), output_dir=str(source))


def estimate_pose_with_densefusion(project="~/Dataset/forcemap_evaluation", name="20240131_191917"):
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


#####
##### Image-based evaluation
#####


# def compute_3d_position(mask, width=640, height=480):
#     xs = np.arange(width) * mask
#     x = np.average(xs[xs != 0])
#     ys = np.arange(height).reshape((height, 1)) * mask
#     y = np.average(ys[ys != 0])
#     return x, y


# def compute_3d_position(mask, points):
#     msk = mask[:, :, np.newaxis]
#     masked_points = points * msk
#     n_valid_points = np.count_nonzero(masked_points[:, :, 2])
#     return [np.sum(masked_points[:, :, i]) / n_valid_points for i in range(3)]


def compute_3d_position(mask, points, outlier_removal="dbscan", n_components=1, n_sigma=1.5, eps=0.01, min_samples=10):
    # outlier_removal := 'gm' | 'std'
    masked_points = apply_mask(points, mask)

    assert outlier_removal == "gm" or outlier_removal == "std" or outlier_removal == "dbscan"
    if outlier_removal == "std":
        m = masked_points.mean(axis=0)
        s = masked_points.std(axis=0)
        lb = m - n_sigma * s
        ub = m + n_sigma * s
        points_without_outliers = masked_points[np.all(lb < masked_points, axis=1)]
        points_without_outliers = points_without_outliers[np.all(points_without_outliers < ub, axis=1)]
        ret = np.average(points_without_outliers, axis=0)
    elif outlier_removal == "gm":
        g = mixture.GaussianMixture(n_components=n_components, covariance_type="full")
        g.fit(masked_points)
        ret = g.means_[np.argmax(g.weights_)]
    else:
        # https://scikit-learn.org/stable/modules/clustering.html#dbscan
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(masked_points)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise = list(labels).count(-1)

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        largest_cluster = np.zeros((1, 3))
        for k in unique_labels:
            if k != -1:
                class_member_mask = labels == k
                cluster = masked_points[class_member_mask & core_samples_mask]
                if cluster.shape[0] > largest_cluster.shape[0]:
                    largest_cluster = cluster
        ret = largest_cluster.mean(axis=0)
    return ret


def image_based_tracking(project="~/Dataset/forcemap_evaluation/01_GAFS_1"):
    p = Path(project).expanduser()
    trajectories = {}
    poly_files = sorted(glob.glob(str(p / "yolo_out" / "labels" / "*-color.txt")))

    for frame_number, poly_file in enumerate(poly_files):
        n = int(Path(poly_file).stem.replace("-color", ""))
        clss, masks, bbs = polygon_to_mask(poly_file, unify_mask=False)
        depth_path = p / f"{n:06d}-depth.png"
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
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


def plot_trajectory(trajs, object_name, z_offset=-0.7, out_file=None):
    traj = trajs[object_name]
    indices = np.array(list(traj.keys()))
    poss, oris = unzip(traj.values())

    ts = indices * 0.033  # convert index to second

    xs, ys, zs = unzip(np.array(poss))
    fig, ax = plt.subplots()
    ax.plot(ts, xs, "o-", label="x")
    ax.plot(ts, ys, "o-", label="y")
    if z_offset == 0.0:
        ax.plot(ts, zs, "o-", label="z")
    else:
        ax.plot(ts, np.array(zs) + z_offset, "o-", label=f"z{z_offset}")
    ax.set_title(object_name)
    ax.set_xlabel("[sec]")
    ax.set_ylabel("[m]")
    ax.legend(loc="upper right")
    if out_file != None:
        of = out_file.parent / (out_file.stem + "_trans.png")
        print(of)
        plt.savefig(of)
    else:
        plt.show()

    qw, qx, qy, qz = unzip([quaternion.as_float_array(ori) for ori in oris])
    fig, ax = plt.subplots()
    ax.plot(ts, qw, "o-", label="qw")
    ax.plot(ts, qx, "o-", label="qx")
    ax.plot(ts, qy, "o-", label="qy")
    ax.plot(ts, qz, "o-", label="qz")
    ax.set_title(object_name)
    ax.set_xlabel("[sec]")
    # ax.set_ylabel("[m]")
    ax.legend(loc="upper right")
    if out_file != None:
        of = out_file.parent / (out_file.stem + "_rot.png")
        print(of)
        plt.savefig(of)
    else:
        plt.show()


def apply_mask(points, mask):
    msk = mask[:, :, np.newaxis]
    points = points * msk
    masked_points = points[np.all(points != 0.0, axis=2)]  # instance mask & valid depth
    return masked_points


def plot_distribution(depth, object_name, frame_no, n_bins=50, range=(0.5, 1.5)):
    h = np.histogram(depth, bins=n_bins, range=range)
    plt.title("depth distribution")
    plt.xlabel("[m]")
    plt.ylabel("number of points")
    plt.title(f"{object_name}, frame={frame_no}")
    plt.grid(True)

    w = h[1][1] - h[1][0]
    plt.bar(h[1][:-1], h[0], width=w, alpha=0.5, color="c")
    # plt.hist(depth, bins=n_bins, alpha=0.5, color="c")
    plt.show()


def depth_distribution(object_name, frame_no, n_bins=50):
    poly_path = f"/home/ryo/Program/yolov5/runs/predict-seg/20240131_193129/labels/{frame_no:06d}-color.txt"
    clss, masks, bbs = polygon_to_mask(poly_path, unify_mask=False)
    depth_path = f"/home/ryo/Dataset/forcemap_evaluation/20240131_193129/{frame_no:06d}-depth.png"
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    points = back_project(depth_img)
    j = [i for i, c in enumerate(clss) if label_names[c] == object_name][0]  # find the mask of object_name
    msk = masks[j]
    masked_points = apply_mask(points, msk)
    depth = masked_points[:, 2]
    plot_distribution(depth, object_name, frame_no, n_bins=n_bins)
    return masked_points, depth_img


# ffmpeg -r 30 -i %06d-color.jpg -vcodec libx264 -pix_fmt yuv420p out.mp4

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str)
parser.add_argument("--conf_thres", type=float, default=0.15)
args = parser.parse_args()

if __name__ == "__main__":
    gen_bb_and_mask_with_YOLO(args.project, args.conf_thres)
