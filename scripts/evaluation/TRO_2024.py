import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import norm
from tracking import *

# def eval_traj(traj):
#     a = np.array(list(traj.values()))
#     if a.shape[0] < 5:
#         return 0.0
#     # return np.max(norm(a - a[0], axis=1, ord=2))  # outlier removal is needed
#     return norm(a[-1] - a[4], ord=2)

# def eval_trajectories_for_all_scenes(project):
#     p = Path(project)
#     scene_dirs = sorted(glob.glob(str(p / "04_IFS_1")))

#     for scene_dir in scene_dirs:
#         print(scene_dir)
#         trajs = pd.read_pickle(str(Path(scene_dir) / "yolo_clustering" / "trajectories.pkl"))
#         for name, traj in trajs.items():
#             print(scene_dir, name, eval_traj(traj))


def print_latex_table(scores, unit="cm"):
    scene_numbers = [1, 2, 3, 4, 5, 6, 8]
    methods = ["UP", "IFS", "GAFS"]
    scale = 1.0
    if unit == "cm" or unit == "cm/s" or unit == "$cm/s^2$":
        scale = 100.0

    def print_header():
        for scene in scene_numbers:
            print(f" & scene{scene} ({unit})$\downarrow$", end="")
        print(f" & All ({unit})$\downarrow$")
        print("\\\\ \hline\hline")

    def print_latex_table_line(method):
        print(method, end="")
        means = []
        stds = []
        for scene in scene_numbers:
            v = np.array(scores[f"{scene:02d}_{method}"])
            if is_best(scene, method):
                print(f" & {{\\bf {v.mean()*scale:.2f} $\pm$ {v.std()*scale:.2f}}}", end="")
            else:
                print(f" & {v.mean()*scale:.2f} $\pm$ {v.std()*scale:.2f}", end="")
            means.append(v.mean())
            stds.append(v.std())
        print(f" & {np.array(means).mean()*scale: .2f} $\pm$ {np.array(stds).mean()*scale:.2f}", end="")
        print("\\\\ \hline")

    def is_best(scene, method):
        vals = [(m, np.average(scores[f"{scene:02d}_{m}"])) for m in methods]
        best_method = sorted(vals, key=lambda x: x[1])[0][0]
        return best_method == method

    print_header()
    for method in methods:
        print_latex_table_line(method)


def image_based_tracking_for_all_scenes(project):
    p = Path(project)
    scene_dirs = glob.glob(str(p / "*"))

    for scene_dir in scene_dirs:
        print(scene_dir)
        trajs = image_based_tracking(project=scene_dir)

        out_dir = Path(scene_dir) / "yolo_clustering"
        if not out_dir.exists():
            out_dir.mkdir()
        pd.to_pickle(trajs, out_dir / "trajectories.pkl")

        for target_name in trajs.keys():
            plot_trajectory(trajs, target_name, out_file=out_dir / f"{target_name}.png")


def plot_fpose_trajectories_for_all_scenes(project):
    p = Path(project)
    scene_dirs = glob.glob(str(p / "*"))

    for scene_dir in scene_dirs:
        print(scene_dir)
        sd = scene_dir.split("/")[-1]
        scene, method, trial = sd.split("_")
        trial = int(trial)
        trajs = fpose_load_trajectories(project, scene, method, trial)

        out_dir = Path(scene_dir) / "fpose_out"
        if not out_dir.exists():
            out_dir.mkdir()

        for target_name in trajs.keys():
            plot_trajectory(trajs, target_name, z_offset=0.0, out_file=out_dir / f"{target_name}.png")


def gen_fpose_anims_for_all_scenes(project):
    p = Path(project)
    scene_dirs = glob.glob(str(p / "*"))

    for scene_dir in scene_dirs:
        print(scene_dir)
        fpose_img_dir = Path(scene_dir) / "track_vis"
        traj_dirs = glob.glob(str(fpose_img_dir / "*"))
        for traj_dir in traj_dirs:
            print(traj_dir)
            img_files = sorted(glob.glob(traj_dir + "/*"))
            images = [Image.open(img_file) for img_file in img_files]
            target = traj_dir.split("/")[-1]
            images[0].save(
                str(Path(scene_dir) / f"fpose_out/{target}.gif"),
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=33,
                loop=0,
            )


start_end_displ = {
    "01_GAFS": [0.035, 0.032, 0.031],
    "01_IFS": [0.061, 0.034, 0.064],
    "01_UP": [0.036, 0.108, 0.064],
    "02_GAFS": [0.054, 0.043, 0.049],
    "02_IFS": [0.059, 0.054, 0.062],
    "02_UP": [0.087, 0.049, 0.049],
    "03_GAFS": [0.015, 0.009, 0.008],
    "03_IFS": [0.007, 0.008, 0.009],
    "03_UP": [0.009, 0.009, 0.009],
    "04_GAFS": [0.041, 0.03, 0.045],
    "04_IFS": [0.055, 0.032, 0.061],
    "04_UP": [0.036, 0.038, 0.033],
    "05_GAFS": [0.010, 0.026, 0.020],
    "05_IFS": [0.025, 0.114, 0.010],
    "05_UP": [0.021, 0.014, 0.016],
    "06_GAFS": [0.081, 0.049, 0.045],
    "06_IFS": [0.112, 0.047, 0.144],
    "06_UP": [0.097, 0.156, 0.177],
    "08_GAFS": [0.067, 0.049, 0.061],
    "08_IFS": [0.037, 0.149, 0.043],
    "08_UP": [0.061, 0.109, 0.128],
}

scenes = {
    "01_UP": [(50, 110), (40, 100), (25, 85)],
    "01_IFS": [(85, 160), (20, 70), (25, 85)],
    "01_GAFS": [(70, 140), (35, 130), (30, 100)],
    "02_UP": [(70, 120), (35, 90), (30, 80)],
    "02_IFS": [(25, 85), (20, 70), (20, 80)],
    "02_GAFS": [(40, 90), (30, 80), (40, 95)],
    "03_UP": [(15, 68), (20, 75), (20, 75)],
    "03_IFS": [(20, 120), (20, 85), (30, 90)],
    "03_GAFS": [(25, 80), (20, 75), (15, 80)],
    "04_UP": [(25, 80), (15, 80), (15, 70)],
    "04_IFS": [(25, 100), (20, 100), (20, 100)],
    "04_GAFS": [(40, 130), (30, 115), (30, 110)],
    "05_UP": [(20, 90), (15, 85), (15, 80)],
    "05_IFS": [(25, 105), (25, 105), (20, 125)],
    "05_GAFS": [(30, 110), (40, 110), (25, 90)],
    "06_UP": [(25, 130), (15, 155), (15, 160)],
    "06_IFS": [(25, 160), (20, 95), (20, 160)],
    "06_GAFS": [(40, 228), (25, 165), (25, 150)],
    "08_UP": [(15, 75), (15, 70), (15, 75)],
    "08_IFS": [(30, 95), (15, 70), (20, 75)],
    "08_GAFS": [(20, 120), (40, 100), (20, 75)],
}


def bag_to_images_for_all(bag_dir, output_dir):
    # No anaconda env
    for scene, frames in scenes.items():
        for i, (start_index, end_index) in enumerate(frames):
            bag_file = f"{bag_dir}/{scene}_{i+1}.bag"
            cmd = f"python bag_to_images_rs.py --bag_file {bag_file} --output_dir {output_dir} --start_index {start_index} --end_index {end_index}"
            print(cmd)
            subprocess.run(cmd, shell=True)


def yolo_tracking_for_all(project_dir):
    for scene, frames in scenes.items():
        for i, (_, _) in enumerate(frames):
            project = f"{project_dir}/{scene}_{i+1}"
            cmd = f"python tracking.py --project {project}"
            print(cmd)
            subprocess.run(cmd, shell=True)


def generate_YOLO_movies_for_all(project_dir):
    for scene, frames in scenes.items():
        for i, (_, _) in enumerate(frames):
            project = f"{project_dir}/{scene}_{i+1}/yolo_out"
            cmd = f"ffmpeg -r 20 -i {project}/%06d-color.jpg -vcodec libx264 {project}/yolo_out.mp4"
            print(cmd)
            subprocess.run(cmd, shell=True)


def generate_FPOSE_movies_for_all(project_dir):
    for scene, frames in scenes.items():
        for i, (_, _) in enumerate(frames):
            project = f"{project_dir}/{scene}_{i+1}/yolo_out"
            cmd = f"ffmpeg -r 20 -i {project}/%06d-color.jpg -vcodec libx264 {project}/yolo_out.mp4"
            print(cmd)
            subprocess.run(cmd, shell=True)


# surrounding objects
scene_and_targets = {
    "01": ["061_foam_brick"],
    "02": ["007_tuna_fish_can"],
    "03": ["009_gelatin_box"],
    # "04": ["051_large_clamp", "052_extra_large_clamp", "061_foam_brick"],
    "04": ["051_large_clamp", "061_foam_brick"],
    "05": ["009_gelatin_box"],
    "06": ["005_tomato_soup_can"],
    "08": ["007_tuna_fish_can", "009_gelatin_box"],
}

import glob
import re

import quaternion
import scipy.linalg


def fpose_load_trajectories(project_dir, scene="01", method="GAFS", trial=1):
    trajs = {}
    targets = scene_and_targets[scene]
    for target in targets:
        traj = {}
        files = sorted(glob.glob(f"{project_dir}/{scene}_{method}_{trial}/ob_in_cam/*{target}.bb.txt"))
        last_q = np.zeros(4)
        for file in files:
            frame_num = int(re.search("\d\d\d\d\d\d", file)[0])
            m = np.loadtxt(file)
            q = quaternion.from_rotation_matrix(m[:3, :3])
            qq = quaternion.as_float_array(q)
            d1 = scipy.linalg.norm(last_q - qq)
            d2 = scipy.linalg.norm(last_q + qq)
            if d1 > d2:
                q = -q
            last_q = quaternion.as_float_array(q)
            waypoint = pose = m[:3, 3], q
            traj[frame_num] = waypoint
        trajs[target] = traj
    return trajs


def eval_trajectories_for_all_scenes(project_dir, eval_fn):
    scores = {}
    methods = ["GAFS", "IFS", "UP"]
    for scene in scene_and_targets.keys():
        for method in methods:
            s = []
            key = f"{scene}_{method}"
            print(key)
            for trial in range(1, 4):
                trajs = fpose_load_trajectories(project_dir, scene, method, trial)
                s.append(eval_fn(trajs))
            scores[key] = s
    return scores


def total_distance_translation(trajs):
    total_distance = 0.0
    for traj in trajs.values():
        frames = sorted(traj.keys())
        for i in range(len(frames) - 1):
            wp1 = traj[frames[i]]
            wp2 = traj[frames[i + 1]]
            pos1 = wp1[0]
            pos2 = wp2[0]
            total_distance += scipy.linalg.norm(pos1 - pos2, ord=2)
    return total_distance


def max_velocity_translation(trajs, dt=0.033):
    max_velocity = 0.0
    for traj in trajs.values():
        frames = sorted(traj.keys())
        for i in range(len(frames) - 1):
            wp1 = traj[frames[i]]
            wp2 = traj[frames[i + 1]]
            pos1 = wp1[0]
            pos2 = wp2[0]
            v = scipy.linalg.norm(pos1 - pos2, ord=2) / dt
            if v > max_velocity:
                max_velocity = v
    return max_velocity


def max_acceleration_translation(trajs, dt=0.033):
    max_acceleration = 0.0
    for traj in trajs.values():
        frames = sorted(traj.keys())
        for i in range(1, len(frames) - 1):
            wp1 = traj[frames[i - 1]]
            wp2 = traj[frames[i]]
            wp3 = traj[frames[i + 1]]
            pos1 = wp1[0]
            pos2 = wp2[0]
            pos3 = wp3[0]
            a = scipy.linalg.norm(pos1 - 2.0 * pos2 + pos3, ord=2) / dt**2
            if a > max_acceleration:
                max_acceleration = a
    return max_acceleration


def rot_distance(q1, q2, axis=None):
    if axis == None:
        d = 2 * np.arccos(np.dot(quaternion.as_float_array(q1), quaternion.as_float_array(q2)))
    else:
        axes = {"x": 0, "y": 1, "z": 2}
        axis_idx = axes[axis]
        v1 = quaternion.as_rotation_matrix(q1)[:, axis_idx]
        v2 = quaternion.as_rotation_matrix(q2)[:, axis_idx]
        d = np.arccos(np.dot(v1, v2))
    return d


def get_axis(target):
    if target == "007_tuna_fish_can":
        axis = "x"
    elif target == "005_tomato_soup_can":
        axis = "z"
    else:
        axis = None
    return axis


def total_distance_rotation(trajs):
    total_distance = 0.0
    for target, traj in trajs.items():
        frames = sorted(traj.keys())
        for i in range(len(frames) - 1):
            wp1 = traj[frames[i]]
            wp2 = traj[frames[i + 1]]
            q1 = wp1[1]
            q2 = wp2[1]
            total_distance += rot_distance(q1, q2, axis=get_axis(target))
    return total_distance


def max_velocity_rotation(trajs, dt=0.033):
    max_velocity = 0.0
    for target, traj in trajs.items():
        frames = sorted(traj.keys())
        for i in range(len(frames) - 1):
            wp1 = traj[frames[i]]
            wp2 = traj[frames[i + 1]]
            q1 = wp1[1]
            q2 = wp2[1]
            omega = rot_distance(q1, q2, axis=get_axis(target)) / dt
            if omega > max_velocity:
                max_velocity = omega
    return max_velocity


project_dir = "/home/ryo/Dataset/forcemap_evaluation"

if __name__ == "__main__":
    # bag_to_images_for_all(bag_dir="/home/ryo/Dataset/bags/exp", project_dir)
    # yolo_tracking_for_all(project_dir)
    # image_based_tracking_for_all_scenes(project_dir)

    # foundationpose_eval_traj_for_all_scenes(project_dir)

    # plot_fpose_trajectories_for_all_scenes(project_dir)
    gen_fpose_anims_for_all_scenes(project_dir)

    # eval_traj_for_all_scenes("/home/ryo/Dataset/forcemap_evaluation/")  # YOLO + backprojection + pointcloud clustering
    # print_latex_table(start_end_displ)
