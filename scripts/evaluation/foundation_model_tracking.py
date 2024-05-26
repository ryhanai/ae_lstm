import subprocess

# scene_and_targets = {
#     "01": ["009_gelatin_box", "061_foam_brick"],
#     "02": ["009_gelatin_box", "007_tuna_fish_can"],
#     "03": ["009_gelatin_box"],
#     "04": ["011_banana", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"],
#     "05": ["009_gelatin_box", "037_scissors"],
#     "06": ["005_tomato_soup_can", "051_large_clamp", "052_extra_large_clamp"],
#     "08": ["007_tuna_fish_can", "009_gelatin_box", "051_large_clamp", "052_extra_large_clamp"],
# }
scene_and_targets = {
    "04": ["011_banana", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"],
}


def foundation_pose_tracking_for_all(project_dir):
    for scene, targets in scene_and_targets.items():
        for method in ["GAFS", "IFS", "UP"]:
            for i in range(0, 3):
                scene_method = f"{scene}_{method}_{i+1}"
                for target in targets:
                    cmd = f"python /home/ryo/Program/FoundationPose/run_demo.py --test_scene {scene_method} --target {target} --debug 2"
                    print(cmd)
                    subprocess.run(cmd, shell=True)


project_dir = "/home/ryo/Dataset/forcemap_evaluation"


if __name__ == "__main__":
    foundation_pose_tracking_for_all(project_dir)
