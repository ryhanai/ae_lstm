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


if __name__ == "__main__":
    # image_based_tracking_for_all_scenes("/home/ryo/Dataset/forcemap_evaluation/")
    # eval_traj_for_all_scenes("/home/ryo/Dataset/forcemap_evaluation/")
    print_latex_table(start_end_displ)
