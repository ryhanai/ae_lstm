import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import norm
from tracking import *


def eval_traj(traj):
    a = np.array(list(traj.values()))
    if a.shape[0] < 5:
        return 0.0
    # return np.max(norm(a - a[0], axis=1, ord=2))  # outlier removal is needed
    return norm(a[-1] - a[4], ord=2)


def eval_traj_for_all_scenes(project):
    p = Path(project)
    scene_dirs = sorted(glob.glob(str(p / "04_IFS_1")))

    for scene_dir in scene_dirs:
        print(scene_dir)
        trajs = pd.read_pickle(str(Path(scene_dir) / "yolo_clustering" / "trajectories.pkl"))
        for name, traj in trajs.items():
            print(scene_dir, name, eval_traj(traj))


def print_latex_table(scores):
    def print_latex_table_line(scores, method):
        print(method, end="")
        means = []
        stds = []
        for scene in [1, 2, 3, 4, 5, 6, 8]:
            v = np.array(scores[f"{scene:02d}_{method}"])
            print(f" & {v.mean()*100:.2f} $\pm$ {v.std()*100:.2f}", end="")
            means.append(v.mean())
            stds.append(v.std())
        print(f" & {np.array(means).mean()*100: .2f} $\pm$ {np.array(stds).mean()*100:.2f}", end="")
        print("\\\\ \hline")

    for method in ["UP", "IFS", "GAFS"]:
        print_latex_table_line(scores, method)


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
    "04_GAFS": [0.041, 0.03, 0.045],  ## 04_GAFS_2 is strange
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


def generate_movies_for_all(project_dir):
    for scene, frames in scenes.items():
        for i, (_, _) in enumerate(frames):
            project = f"{project_dir}/{scene}_{i+1}/yolo_out"
            cmd = f"ffmpeg -r 20 -i {project}/%06d-color.jpg -vcodec libx264 {project}/yolo_out.mp4"
            print(cmd)
            subprocess.run(cmd, shell=True)


project_dir = "/home/ryo/Dataset/forcemap_evaluation"

if __name__ == "__main__":
    # bag_to_images_for_all(bag_dir="/home/ryo/Dataset/bags/exp", project_dir)
    # yolo_tracking_for_all(project_dir)
    # image_based_tracking_for_all_scenes(project_dir)
    generate_movies_for_all(project_dir)
    # eval_traj_for_all_scenes("/home/ryo/Dataset/forcemap_evaluation/")  # YOLO + backprojection + pointcloud clustering
    # print_latex_table(start_end_displ)
