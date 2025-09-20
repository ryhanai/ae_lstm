import re
from pathlib import Path

import numpy as np
import transforms3d as tf
from working.test_lerobot_data2 import LeRobotRecorder

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
import cv2
from aist_sb_ur5e.task import ConveniPickupTask
from dataset.object_loader import ObjectInfo
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper
from sim.controller.kinematics_solver import KinematicsSolver  # for UR5e
from sim.generated_grasps import *
from sim.motion_planning import trapezoidal_trajectory
from abc import ABC, abstractmethod
import numpy.typing as npt
import json_numpy
import requests
import time

json_numpy.patch()  # default json doesn't support the serialization of ndarray



import omni.ui as ui
# Create Window
hud_window = ui.Window("Text Instruction", width=400, height=50)
with hud_window.frame:
    with ui.VStack():
        hud_label = ui.Label("", style={"color": 0xff00ffff, "font_size": 20})  # RGBA

def update_hud_text(new_text: str):
    hud_label.text = new_text


message_id = 0
image_size = [120, 160]  # [height, width]

oi = ObjectInfo()


def get_object_center(task, object_name : str):
    obj_pos, obj_quat = get_product_world_pose(task, object_name)
    product_name = re.findall("(.*)_\d+_\d+", object_name)[0]  # product instance name -> product (class) name
    return transform_to_centroid(product_name, obj_pos, obj_quat)


HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()


my_world: World = World(stage_units_in_meters=1.0)
task = ConveniPickupTask(static_path=HELLO_ISAAC_ROOT / "aist_sb_ur5e/static")
my_world.add_task(task=task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(name=task.get_params()["robot_names"]["value"][0])
target: XFormPrim = my_world.scene.get_object(name=task.get_params()["x_form_prim_names"]["value"][0])
gripper: ParallelGripper = task.get_params()["gripper"]["value"]

print(f"PD GAIN: {ur5e._articulation_controller.get_gains()}")
print(f"MAX EFFORTS: {ur5e._articulation_controller.get_max_efforts()}")
ur5e._articulation_controller.set_max_efforts([1e6] * 6 + [0, 100, 0, 100, 0, 0])
# stiffnesses = [150000.0] * 6 + [0, 5e4, 0, 5e4, 0, 0]
# dampings = [72500.0] * 6 + [0, 4e4, 0, 4e4, 0, 0]
stiffnesses = [60000.0] * 6 + [0, 5e4, 0, 5e4, 0, 0]
dampings = [2500.0] * 6 + [0, 4e4, 0, 4e4, 0, 0]
ur5e._articulation_controller.set_gains(kps=stiffnesses, kds=dampings)
print(f"PD GAIN (2): {ur5e._articulation_controller.get_gains()}")
print(f"MAX EFFORTS (2): {ur5e._articulation_controller.get_max_efforts()}")

ik_solver = KinematicsSolver(robot_articulation=ur5e)
ik_solver._kinematics.set_robot_base_pose(*ur5e.get_world_pose())


def convert_to_joint_angle(gripper_opening_width: float):
    return 0.725 - np.arcsin(gripper_opening_width / 0.21112)


def reset_robot_state():
    js = ur5e.get_joints_default_state()
    q = convert_to_joint_angle(gripper_opening_width=0.12)
    js.positions[7] = q
    js.positions[9] = -q
    ur5e.set_joint_positions(js.positions)


def crop_center_and_resize(img, crop_size=[540, 768], output_size=[240, 320]):
    height, width = crop_size
    cam_height = img.shape[0]
    cam_width = img.shape[1]
    cropped_img = img[
        int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
        int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
    ]
    return cv2.resize(cropped_img, (output_size[1], output_size[0]))


def get_product_world_pose(task, target_name: str):
    for product in task._convenience_store.products:
        if product.name == target_name:
            return product.get_world_pose()  # get_world_pose() returns scalar-first quaternion


def plan_picking_trajectory(task, target_object):
    group, obj_name, obj_id, (init_pos, init_quat), scale = target_object
    target_pose = get_object_center(task, f"{obj_name}_{obj_id[0]}_{obj_id[1]}")
    tp = target_pose[0]
    bb = np.array(group.object_dimension)
    slide = bb[0] / 4.0
    dz = 0.02
    wps = [
        (np.array([0.45, 0.3, 1.35]), 0.015), # initial position
        (tp + [bb[0] / 2 + 0.025, 0, dz + 0.05], 0.015),  # back and up of the object
        (tp + [bb[0] / 2 + 0.025, 0, dz], 0.015), # back of the object
        (tp + [bb[0] / 2 + 0.012, 0, dz], 0.015), # back of and touching the object
        (tp + [bb[0] / 2 + 0.012 - slide, 0, dz + 0.005], 0.015), # slide a little the object
        (tp + [bb[0] / 2 + 0.012 - slide + 0.04, 0, dz + 0.05], 0.015),  # back and up of the object
        # tp + [-bb[0] / 2 - slide - 0.04, 0, 0.05],
        # tp + [-bb[0] / 2 - slide - 0.04, 0, 0],
    ]

    wps = [(p + np.array([0, 0, 0.2]), convert_to_joint_angle(w)) for p, w in wps]  # offset between "the middle of finger tips" and 'tool0'
    return wps


class TaskEnvironment:
    def __init__(self, task, recorder=None):
        self._task = task
        self._recorder = recorder

    def reset(self):
        self._target_object = task._convenience_store.display_products()
        self._task_description = f"pick a {self._target_object[1]}"

        update_hud_text(self._task_description)

        if not (self._recorder is None):
            self._recorder.new_episode(task_description=self._task_description)
        return self._target_object

    def get_observation(self):
        qpos = ur5e.get_joint_positions()[:7]
        qvel = np.zeros(7)  # not yet used
        effort = np.zeros(7)  # not yet used
        image_left = crop_center_and_resize(task._cameras[0].get_rgb(), output_size=image_size)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2BGR)
        image_right = crop_center_and_resize(task._cameras[1].get_rgb(), output_size=image_size)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)
        return qpos, qvel, effort, image_left, image_right, self._task_description

    def step(self, action, observation, end_flag):
        if end_flag:
            if self.success_condition_satisfied():
                print("[TASK SUCCESS]: goal reached")
                if not (self._recorder is None):
                    self._recorder.save_episode()
            else:
                print("[TASK FAILURE]: success condition not satisfied")

        else:
            arm_action = list(action[:6]) + [None] * 6
            ur5e.get_articulation_controller().apply_action(ArticulationAction(arm_action))
            gripper_action = [-action[6], action[6]]

            # print(f'GRIPPER GOAL={gripper_action}')
            gripper.apply_action(ArticulationAction(joint_positions=gripper_action))

            if not (self._recorder is None):
                qpos, qvel, effort, image_left, image_right, task_description = observation

                self._recorder.step(
                    qpos=qpos,
                    qvel=qvel,
                    effort=effort,
                    action=action,
                    image_left=image_left,
                    image_right=image_right,
                )

    def save_initial_state(self):
        self._initial_poses = {}
        for product in self._task._convenience_store.products:
            obj_pos, obj_quat = product.get_world_pose()
            product_class_name = re.findall("(.*)_\d+_\d+", product.name)[
                0
            ]  # product instance name -> product (class) name
            try:
                self._initial_poses[product.name] = transform_to_centroid(product_class_name, obj_pos, obj_quat)
            except:
                pass

    def success_condition_satisfied(self):
        is_success = True

        # target object moved as is expected
        group, obj_name, obj_id, (init_pos, init_quat), scale = self._target_object
        target_name = f'{obj_name}_{obj_id[0]}_{obj_id[1]}'

        # bb = np.array(group.object_dimension)
        # current_position = get_object_center(task, target_name)[0]
        # initial_position = initial_poses[target_name][0]
        # if np.linalg.norm(current_position - initial_position) > 0.01:
        #     is_success = False
        #     print(f'{target_name} moved')

        # other objects were not moved
        for p in self._task._convenience_store.products:
            if p.name != target_name:
                current_position = get_object_center(task, p.name)[0]
                initial_position = self._initial_poses[p.name][0]
                if np.linalg.norm(current_position - initial_position) > 0.01:
                    print(f'{p.name} moved, initial={initial_position}, current={current_position}')
                    is_success = False
                    break

        return is_success


class Policy(ABC):
    @abstractmethod
    def reset(self, target_object):
        pass

    @abstractmethod
    def get_action(self, observation) -> tuple[npt.NDArray[np.float32], bool]:
        pass


class LearningBasedPolicy(Policy):
    """
    Inference server must be started in another process
    $ python inference_service.py --server --http-server --port 8000 --model_path /data2/SB_gr00t/model/path --denoising-steps 4
    """

    def __init__(self):
        pass

    def reset(self, target_object):
        pass

    def get_action(self, observation) -> tuple[npt.NDArray[np.float32], bool]:
        qpos, qvel, effort, image_left, image_right, task_description = observation
        x = {
            'state.qpos': qpos[np.newaxis, :],
            'video.left_view': image_left[np.newaxis, :],
            'video.right_view': image_right[np.newaxis, :],
            'annotation.human.task_description': [task_description],
        }

        t = time.time()
        response = requests.post(
            "http://0.0.0.0:8000/act",
            # "http://159.223.171.199:44989/act",   # Bore tunnel
            json={"observation": x},
        )
        print(f"used time {time.time() - t}")
        y = response.json()        
        print(f'ACTION={y}')        
        action = y['action.qpos'][15].copy()  # response value is immutable
        return action, False


class SimpleScriptedPolicy(Policy):
    def __init__(self):
        self._vel_range = [0.01, 0.05]

    def reset(self, target_object):
        """
        target object can be replaced with a language instruction
        """
        self._target_object = target_object
        self._waypoints = plan_picking_trajectory(task, target_object)
        self._current_trajectory = None

    def set_next_waypoint(self, current_position):
        arm_goal, gripper_goal = self._waypoints.pop(0)
        ts, xs, vs, as_ = trapezoidal_trajectory(current_position, 
                                                 arm_goal,
                                                 vmax=0.3,
                                                 amax=1.0,
                                                 dt=1/60.,
                                                 )
        self._current_trajectory = xs
        self._current_gripper_goal = gripper_goal
        self._t = 0

    def get_action(self, observation):
        qpos, qvel, effort, image_left, image_right, task_description = observation
        arm_joint_positions = qpos[:6]
        gripper_joint_position = qpos[6]
        current_position, current_quat = ur5e._kinematics.compute_forward_kinematics("tool0", arm_joint_positions)

        if self._current_trajectory is None:
            self.set_next_waypoint(current_position)

        arm_goal = self._current_trajectory[-1]
        gripper_goal = self._current_gripper_goal
        
        if np.linalg.norm(arm_goal - current_position) < 0.005:
            # and np.abs(gripper_goal - gripper_joint_position) < 0.05:
            print("[POLICY]: current goal reached")
            if len(self._waypoints) == 0:
                return qpos, True
            else:
                self.set_next_waypoint(current_position)

        if self._t < self._current_trajectory.shape[0]:
            target_position = self._current_trajectory[self._t]
        else:
            # print(f'DISTANCE left: {np.linalg.norm(arm_goal - current_position)}')
            target_position = self._current_trajectory[-1]

        self._t += 1
        target_orientation = tf.quaternions.mat2quat(tf.euler.euler2mat(0, np.pi, 0))

        position_tolerance = 0.001
        orientation_tolerance = 0.01
        ik_action, ik_success = ik_solver.compute_inverse_kinematics(
            target_position, target_orientation, position_tolerance, orientation_tolerance
        )

        arm_action = ik_action.joint_positions[:6]
        action = np.append(arm_action, -gripper_goal)

        if ik_success:
            if np.allclose(arm_joint_positions, arm_action, atol=0.4):
                return action, False
            else:
                # IK solution is obtained, but it is not good
                print("Too large difference in arm joint positions!")
                return qpos, True
        else:
            print("IK failed!")
            return qpos, True


def main(max_episode_steps=200):
    env = TaskEnvironment(task, recorder=LeRobotRecorder())
    policy = SimpleScriptedPolicy()
    policy = LearningBasedPolicy()
    end_flag = True

    while simulation_app.is_running():
        my_world.step(render=True)

        if end_flag:
            frame_number = 0
            reset_robot_state()
            target_object = env.reset()
            policy.reset(target_object)
            end_flag = False

        if frame_number == 4:  # It takes some time till the renderer's output becomes stable
            env.save_initial_state()

        if my_world.is_playing() and frame_number > 4:
            obs = env.get_observation()
            action, end_flag = policy.get_action(obs)
            env.step(action, obs, end_flag)

        if frame_number >= max_episode_steps:
            print("Episode timed out")
            end_flag = True

        frame_number += 1


main()
simulation_app.close()


# class Recorder:
#     def __init__(
#         self,
#         task,
#         output_directory="picking_experiment_results",
#     ):
#         self._task = task
#         self._data_dir = Path(output_directory)

#     def new_episode(self, name):
#         self._frameNo = 0
#         self._episode_name = name
#         p = self._data_dir / self._episode_name
#         if p.is_dir():
#             raise FileExistsError(f"Output of episode {self._episode_name} already exists in {self._data_dir}.")
#         p.mkdir(parents=True)

#     def save_robot_state(self):
#         robot_state_path = self._data_dir / self._episode_name / f"robot_state{self._frameNo:05}.pkl"
#         pd.to_pickle(self._task._ur5e.get_joint_positions(), robot_state_path)

#     def save_product_state(self):
#         bin_state = []
#         for product in self._task.get_active_products():
#             p, o = product.get_world_pose()
#             o = np.roll(o, 1)  # to scalar first quaternion
#             bin_state.append((product.name, (p, o)))

#         bin_state_path = self._data_dir / self._episode_name / f"bin_state{self._frameNo:05}.pkl"
#         pd.to_pickle(bin_state, bin_state_path)

#     def save_contact_state(self):
#         contact_positions = []
#         contact_normals = []
#         impulse_values = []
#         contacting_objects = []
#         forces = {}

#         for o in task.get_active_products():
#             cs = task._env._product_sensors[o.name]
#             current_frame = cs.get_current_frame()

#             forces[o.name] = current_frame["force"]

#             if current_frame["in_contact"]:
#                 for contact in current_frame["contacts"]:
#                     objectA = contact["body0"]
#                     objectB = contact["body1"]
#                     contact_positions.append(contact["position"])
#                     contact_normals.append(contact["normal"])
#                     impulse_values.append(np.linalg.norm(contact["impulse"]))
#                     contacting_objects.append((objectA, objectB))
#                     # print(f' {objectA}, {objectB}, {contact["impulse"]}')

#         pd.to_pickle(
#             (contact_positions, impulse_values, contacting_objects, contact_normals, forces),
#             self._data_dir / self._episode_name / f"contact_raw_data{self._frameNo:05d}.pkl",
#         )

#     def save_image(self):
#         rgb = self._task._cameras[1].get_rgb()  # side-view camera
#         rgb_path = self._data_dir / self._episode_name / f"rgb{self._frameNo:05}.jpg"
#         output_rgb = crop_center_and_resize(rgb)
#         cv2.imwrite(str(rgb_path), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

#     def save(self):
#         self.save_robot_state()
#         self.save_product_state()
#         self.save_contact_state()
#         self.save_image()
#         self._frameNo += 1