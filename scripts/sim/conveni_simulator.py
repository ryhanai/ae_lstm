from operator import itemgetter
from pathlib import Path
import pandas as pd
# from pyrfc3339 import generate
import transforms3d as tf
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
import cv2
from aist_sb_ur5e.controller import RMPFlowController
from aist_sb_ur5e.controller import KeyboardController
from dataset.object_loader import ObjectInfo
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper

from aist_sb_ur5e.task import ConveniPickupTask
from sim.controller.kinematics_solver import KinematicsSolver  # for UR5e


message_id = 0

oi = ObjectInfo()

def get_product_world_pose(task, target_name: str):
    for product in task._env.products:
        if product.name == target_name:
            return product.get_world_pose()  # get_world_pose() returns scalar-first quaternion


def gripper_pose_in_world_mimo(task, grasp, target_name: str, pregrasp_opening=0.13):
    """
    The depth of grasp is dependent on the pregrasp opening.
    """

    obj_pos, obj_quat = get_product_world_pose(task, target_name)
    Tworld_obj = pos_quat2mat(obj_pos, obj_quat)
    # Tmimo_franka = pos_euler2mat([0, 0, 0.09], [0, 0, np.pi/2])

    ## 2f-140 grasp depth
    offset = 0.002
    grasp_depth = 0.1679 * pregrasp_opening - 0.153827 + offset
    Tmimo_gripper = pos_euler2mat([0, 0, grasp_depth], [0, 0, np.pi/2])
    return pose_from_tf_matrix(Tworld_obj @ grasp @ Tmimo_gripper)  # pose_from_tf_matrix() returns scalar-first quaternion


def get_object_center(task, lifting_target):
    obj_pos, obj_quat = get_product_world_pose(task, lifting_target)
    return transform_to_centroid(lifting_target, obj_pos, obj_quat)


HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()


my_world: World = World(stage_units_in_meters=1.0)
# my_world.get_physics_context().set_gravity(0.0)
# my_world.set_gravity(value=-9.81 / meters_per_unit())

task = ConveniPickupTask(static_path=HELLO_ISAAC_ROOT / "aist_sb_ur5e/static")
my_world.add_task(task=task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(name=task.get_params()["robot_names"]["value"][0])
target: XFormPrim = my_world.scene.get_object(name=task.get_params()["x_form_prim_names"]["value"][0])
gripper: ParallelGripper = task.get_params()["gripper"]["value"]

print(f'PD GAIN: {ur5e._articulation_controller.get_gains()}')
print(f'MAX EFFORTS: {ur5e._articulation_controller.get_max_efforts()}')
ur5e._articulation_controller.set_max_efforts([1e+6]*6 + [0, 100, 0, 100, 0, 0])
stiffnesses = [150000.0]*6 + [0, 5e+4, 0, 5e+4, 0, 0]
dampings = [72500.0]*6 + [0, 4e+4, 0, 4e+4, 0, 0]
ur5e._articulation_controller.set_gains(kps=stiffnesses, kds=dampings)
print(f'PD GAIN (2): {ur5e._articulation_controller.get_gains()}')
print(f'MAX EFFORTS (2): {ur5e._articulation_controller.get_max_efforts()}')

rmpflow_controller = RMPFlowController(
    # name="cspace_controller",
    robot_articulation=ur5e,
    robot_description_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/robot_descriptor.yml"),
    rmpflow_config_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/ur5e_rmpflow_common.yml"),
    urdf_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/urdf/ur5e.urdf"),
)

ik_solver = KinematicsSolver(robot_articulation=ur5e)
ik_solver._kinematics.set_robot_base_pose(*ur5e.get_world_pose())

target_controller = KeyboardController()


def set_joint_positions_UR5e(arm_joint_positions, gripper_joint_position):
    joint_positions = np.zeros(12)
    joint_positions[[6,8,9]] = -gripper_joint_position
    joint_positions[[7,10,11]] = gripper_joint_position
    joint_positions[:6] = arm_joint_positions
    ur5e.set_joint_positions(joint_positions)
    ur5e.set_joint_velocities(np.zeros(12))
    return joint_positions


def convert_to_joint_angle(finger_distance):
    return 0.725 - np.arcsin(finger_distance / 0.21112)


def reset_robot_state():
    js = ur5e.get_joints_default_state()
    ur5e.set_joint_positions(js.positions)


def crop_center_and_resize(img, crop_size=[768,540], output_size=[320,240]):
    width, height = crop_size
    cam_height = img.shape[0]
    cam_width = img.shape[1]
    cropped_img = img[
        int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
        int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
    ]
    return cv2.resize(cropped_img, output_size)


class Recorder:
    def __init__(self, 
                 task,
                 output_directory="picking_experiment_results",
                 ):
        self._task = task
        self._data_dir = Path(output_directory)

    def new_episode(self, name):
        self._frameNo = 0
        self._episode_name = name
        p = self._data_dir / self._episode_name
        if p.is_dir():
            raise FileExistsError(f"Output of episode {self._episode_name} already exists in {self._data_dir}.")
        p.mkdir(parents=True)

    def save_robot_state(self):
        robot_state_path = self._data_dir / self._episode_name / f"robot_state{self._frameNo:05}.pkl"
        pd.to_pickle(self._task._ur5e.get_joint_positions(), robot_state_path)

    def save_product_state(self):
        bin_state = []
        for product in self._task.get_active_products():
            p, o = product.get_world_pose()
            o = np.roll(o, 1)  # to scalar first quaternion
            bin_state.append((product.name, (p, o)))

        bin_state_path = self._data_dir / self._episode_name / f"bin_state{self._frameNo:05}.pkl"
        pd.to_pickle(bin_state, bin_state_path)

    def save_contact_state(self):
        contact_positions = []
        contact_normals = []
        impulse_values = []
        contacting_objects = []
        forces = {}

        for o in task.get_active_products():
            cs = task._env._product_sensors[o.name]
            current_frame = cs.get_current_frame()

            forces[o.name] = current_frame['force']

            if current_frame['in_contact']:
                for contact in current_frame['contacts']:
                    objectA = contact["body0"]
                    objectB = contact["body1"]
                    contact_positions.append(contact["position"])
                    contact_normals.append(contact["normal"])
                    impulse_values.append(np.linalg.norm(contact["impulse"]))
                    contacting_objects.append((objectA, objectB))
                    # print(f' {objectA}, {objectB}, {contact["impulse"]}')

        pd.to_pickle(
            (contact_positions, impulse_values, contacting_objects, contact_normals, forces),
            self._data_dir / self._episode_name / f"contact_raw_data{self._frameNo:05d}.pkl"
        )

    def save_image(self):
        rgb = self._task._cameras[1].get_rgb()  # side-view camera
        rgb_path = self._data_dir / self._episode_name / f"rgb{self._frameNo:05}.jpg"
        output_rgb = crop_center_and_resize(rgb)
        cv2.imwrite(str(rgb_path), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

    def save(self):
        self.save_robot_state()
        self.save_product_state()
        self.save_contact_state()
        self.save_image()
        self._frameNo += 1


def generate_data():
    waypoints = []
    waypoints_reached = False

    while simulation_app.is_running():
        my_world.step(render=True)

        if waypoints == [] and waypoints_reached:
            # reset_robot_state()
            print('change layout')
            target_object = task._convencience_store.display_products()
            waypoints = []

        if my_world.is_playing():
            target_position, target_orientation = target.get_world_pose()
            print(f'Target pose: {target_position}, {target_orientation}')

            next_position, next_orientation = target_controller.forward(
                position=target_position,
                orientation=target_orientation,
            )
            target.set_world_pose(
                position=next_position,
                orientation=next_orientation,
            )

            action: ArticulationAction = rmpflow_controller.forward(
                target_end_effector_position=next_position,
                target_end_effector_orientation=next_orientation,
            )
            ur5e.get_articulation_controller().apply_action(control_actions=action)

            optional_action: str | None = target_controller.get_gripper_action()
            if optional_action is not None:
                gripper_action: ArticulationAction = gripper.forward(optional_action)
                gripper.apply_action(
                    ArticulationAction(
                        joint_positions=itemgetter(7, 9)(gripper_action.joint_positions)
                    )
                )


def do_lifting(end_of_grasping = 120,
               end_of_planned_direction=250,               
               end_of_upward_motion=400,
               end_of_episode = 460,
               lifting_direction=[0., 0., 1.],
               planned_direction_distance=0.10,
               target_lifting_height=0.15,
               recorder=None,               
               ):

    def get_tcp_pose():
        return ur5e._kinematics.compute_forward_kinematics('tool0', ur5e.get_joint_positions()[:6])

    def get_tcp_position():
        return get_tcp_pose()[0]
    
    def distance_from(pos):
        cur_pos = get_tcp_position()
        return np.linalg.norm(cur_pos - pos)

    def move_if_possible(motion_vector, target_orientation):
        target_position, target_orientation = get_tcp_pose()
        target_position += motion_vector
        target_orientation = tf.quaternions.mat2quat(target_orientation)

        position_tolerance = 0.001
        orientation_tolerance = 0.02
        ik_action, success = ik_solver.compute_inverse_kinematics(target_position, target_orientation, position_tolerance, orientation_tolerance)

        if success:
            if np.allclose(ur5e.get_joint_positions()[:6], ik_action.joint_positions[:6], atol=0.5):
                joint_target = list(ik_action.joint_positions) + [None]*6
                ur5e.get_articulation_controller().apply_action(ArticulationAction(joint_target))
                return True
            else:
                joint_target = list(ur5e.get_joint_positions()[:6]) + [None]*6                        
                # print('too large difference in arm joint positions!')
                return False
        else:
            joint_target = list(ur5e.get_joint_positions()[:6]) + [None]*6
            # print('IK failed!')
            return False

    counter = 5
    while simulation_app.is_running() and counter < end_of_episode:
        my_world.step(render=True)

        if my_world.is_playing():
            if 4 < counter <= end_of_grasping:  # do grasp
                gripper_action: ArticulationAction = gripper.forward('close')
                gripper.apply_action(ArticulationAction(joint_positions=itemgetter(7, 9)(gripper_action.joint_positions)))

                # distance = min(dof[0] + dof[1] + counter * 0.0005, 0.14)
                # target_gripper_joint_position = convert_to_joint_angle(distance)
                # target_joint_positions = set_joint_positions(ik_action.joint_positions , target_gripper_joint_position) 

                if counter == end_of_grasping:
                    # print('=> planned direction')
                    pos0, ori0 = get_tcp_pose()

            elif counter <= end_of_planned_direction:  # transport in a planned direction
                if not move_if_possible(motion_vector=0.06 * np.array(lifting_direction), target_orientation=ori0):
                    counter = end_of_planned_direction  # go to the next phase

                if distance_from(pos0) > planned_direction_distance:
                    counter = end_of_planned_direction

            elif counter <= end_of_upward_motion:  # transport upward
                if not move_if_possible(motion_vector=0.06 * np.array([0., 0., 1.]), target_orientation=ori0):
                    counter = end_of_upward_motion  # go to the next phase

                cur_pos = get_tcp_position()
                # print(f'{cur_pos[2]}, {pos0[2]}')
                if cur_pos[2] - pos0[2] > target_lifting_height:
                    print('=> Task succeeded!')                    
                    counter = end_of_upward_motion

            elif counter <= end_of_episode:  # do nothing
                pass

            if (end_of_grasping <= counter) and recorder != None:  # record scenes after grasping
                recorder.save()

            counter += 1


def find_feasible_grasp(scene_idx, lifting_target, pregrasp_pose):
    counter = 0
    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_playing():

            if counter == 0:
                reset_robot_state()

                while True:
                    task.load_bin_state(scene_idx)
                    grasp = grasp_sampler.sample_grasp(lifting_target)
                    g_pos, g_ori = gripper_pose_in_world_mimo(task, grasp, lifting_target)
                    target.set_world_pose(position=g_pos, orientation=g_ori)  # set_world_pose() takes scalar-first quaternion
                    if np.dot(tf.quaternions.quat2mat(g_ori)[:, 2], [0, 0, -1]) < 0.707:
                        continue

                    position_tolerance = 0.002
                    orientation_tolerance = 0.02
                    ik_action, success = ik_solver.compute_inverse_kinematics(g_pos, g_ori, position_tolerance, orientation_tolerance)

                    if not success:
                        continue

                    target_gripper_joint_position = convert_to_joint_angle(pregrasp_pose)
                    target_joint_positions = set_joint_positions_UR5e(ik_action.joint_positions , target_gripper_joint_position) 
                    ur5e.set_joint_velocities(np.zeros(len(target_joint_positions)))
                    break

            elif 0 < counter < 4:  # check collision in the grasp pose
                in_contact = False
                for cs in ur5e._sensors:
                    current_frame = cs.get_current_frame()
                    # print(f'CF: {cs.name}, {current_frame}')
                    if current_frame['in_contact']:
                        for c in current_frame['contacts']:
                            in_contact = True
                            # print(c)
                            # if c['body1'] == '/World/table_surface':
                            #     in_contact = True

                if in_contact:
                    counter = 0
                    continue
                if not np.allclose(target_joint_positions[:6], ur5e.get_joint_positions()[:6], atol=2e-2):
                    print(f'ARM joint poisitions are different from the goal {ur5e.get_joint_positions()[:6]}')
                    counter = 0
                    continue
                if not np.allclose(target_joint_positions[6:], ur5e.get_joint_positions()[6:], atol=8e-2):
                    print(f'GRIPPER joint poisitions are different from the goal {ur5e.get_joint_positions()[6:]}')
                    counter = 0
                    continue

            elif counter == 4:
                return grasp

            counter += 1


def collect_successful_grasps():
    for problem in lifting_problems:
        successful_grasps = []

        print(f'PROBLEM: {problem}')
        if len(problem) == 3:
            scene_idx, lifting_target, pregrasp_pose = problem
        else:
            scene_idx, lifting_target = problem
            pregrasp_pose = 0.13

        while True:
            if len(successful_grasps) >= 3:
                print(f'PROBLEM SOLVED: {(problem[:2], successful_grasps)}')
                break

            grasp = find_feasible_grasp(scene_idx, lifting_target, pregrasp_pose)
            do_lifting()

            if get_product_world_pose(task, lifting_target)[0][2] > 0.85:
                successful_grasps.append((pregrasp_pose, grasp))
                print(f'# of successful grasps = {len(successful_grasps)}')


def run_successful_grasps(lifting_method, tester):
    recorder = Recorder(task, output_directory=f'picking_experiment_results_{lifting_method}')
    success_list = {}

    for problem, successful_grasps in episodes:
        print(f'PROBLEM: {problem}')
        scene_idx, lifting_target = problem

        for grasp_number, (pregrasp_opening, grasp) in enumerate(successful_grasps):
            try:
                recorder.new_episode(name=f'{scene_idx}__{lifting_target}_{grasp_number:03d}__{lifting_method}')
            except FileExistsError as e:
                print(e)
                continue

            reset_robot_state()
            task.load_bin_state(scene_idx)

            g_pos, g_ori = gripper_pose_in_world_mimo(task, grasp, lifting_target)

            for i in range(5):
                my_world.step(render=True)
            img = task._cameras[0].get_rgb()  # capture image from the top-camera
            img = crop_center_and_resize(img)
            cv2.imwrite('/tmp/hoge.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            object_center_pos, object_center_ori = get_object_center(task, lifting_target)
            target.set_world_pose(position=object_center_pos, orientation=object_center_ori)

            position_tolerance = 0.002
            orientation_tolerance = 0.02
            ik_action, success = ik_solver.compute_inverse_kinematics(g_pos, g_ori, position_tolerance, orientation_tolerance)

            target_gripper_joint_position = convert_to_joint_angle(pregrasp_opening)
            target_joint_positions = set_joint_positions_UR5e(ik_action.joint_positions , target_gripper_joint_position) 
            ur5e.set_joint_velocities(np.zeros(len(target_joint_positions)))

            if tester is None:
                direction = [0., 0., 1]
            else:
                predicted_maps, planning_results = tester.predict_from_image(img,
                                                                             object_center_pos,
                                                                             show_result=False,
                                                                             object_radius=0.1)
                print(planning_results)
                direction = planning_results[0]
                if direction[2] < 0.0:
                    direction[2] = 0.0
                    direction /= np.linalg.norm(direction)

            do_lifting(lifting_direction=direction, recorder=recorder)

            if get_product_world_pose(task, lifting_target)[0][2] > 0.85:
                try:
                    success_list[problem].append(grasp_number)
                except:
                    success_list[problem] = [grasp_number]
                print(success_list)


generate_data()
simulation_app.close()
