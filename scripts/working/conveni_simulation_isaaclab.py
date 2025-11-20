##
# 
# $ conda activate neural-npt
# $ python conveni_simulation_isaaclab.py --task Isaac-WRS-TidyUp-HSR-v0 --enable_cameras
#
##


import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Sample code for demonstrating IsaacLabExtendedTasks."
)
parser.add_argument("--task", type=str, default="Isaac-PickUp-ConveniJavaCurry-UR5e-Robotiq2F140-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument("--renderer", type=str, default="RaytracedLighting", help="Renderer to use.")
parser.add_argument("--video", action="store_true", help="Record video.")
parser.add_argument(
    "--video_folder", type=str, default="videos", help="Folder to store videos."
)
parser.add_argument(
    "--video_interval", type=int, default=100, help="Interval to record video."
)
parser.add_argument(
    "--video_length", type=int, default=300, help="Length of the video."
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
# import hsrlab  # noqa: F401
# import hsrlab_tasks  # noqa: F401
import ur5e_conveni.tasks  # noqa: F401


import time
import torch
from isaaclab.sim.utils import bind_physics_material
from isaaclab_tasks.utils import parse_env_cfg

import omni
from omni.kit.viewport.utility import get_active_viewport
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)


def main():
    """Main function."""
    # Create environment configuration
    env_cfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    if args_cli.headless:
        if args_cli.renderer == "PathTracing":
            # Set the renderer to PathTracing
            viewport_api = get_active_viewport()
            viewport_api.set_hd_engine("rtx", "PathTracing")
        elif args_cli.renderer == "RaytracedLighting":
            # Set the renderer to PathTracing
            viewport_api = get_active_viewport()
            viewport_api.set_hd_engine("rtx", "RaytracedLighting")

    if args_cli.video:
        # Record video
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # Reset environment
    env.reset()

    base_env = env.unwrapped  # for some reasons env is OrderEnforcing instance
    robot = base_env.scene["arm"]

    # Simulate physics
    count = 1
    torch.set_printoptions(precision=3, sci_mode=False)
    while simulation_app.is_running():
        # start_time = time.time()
        with torch.inference_mode():
            # Reset
            # if count % 500 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")

            # Sample random actions
            # actions = torch.ones_like(base_env.action_manager.action)  # [x, y, rz, arm_action]
            # actions[:, :3] = 0.0

            actions = torch.zeros((1, 7))

            # Step the environment
            obs, rew, terminated, truncated, info = base_env.step(actions)
            # for name, value in zip(robot.joint_names, obs["proprio"]["joint_pos"].squeeze(0)):
            #     print(f"{name:40s}: {value.item(): .3f}")

            # Update counter
            count += 1
        # print(f"Time taken for step: {time.time() - start_time}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # Close sim app
    simulation_app.close()
