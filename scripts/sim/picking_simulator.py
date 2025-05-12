from operator import itemgetter
from pathlib import Path
import pandas as pd

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
import cv2
import cv_bridge
import rclpy
from aist_sb_ur5e.controller import RMPFlowController
from aist_sb_ur5e.controller import KeyboardController
from dataset.object_loader import ObjectInfo
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper

from sensor_msgs.msg import Image

from sim.task.tabletop_picking_task import TabletopPickingTask
from sim.controller.kinematics_solver import KinematicsSolver  # for UR5e
from visualization_msgs.msg import Marker, MarkerArray
from aist_sb_ur5e.model.factory import create_contact_sensor

from omni.isaac.core import World
from isaacsim.core.utils.rotations import euler_angles_to_quat

# world = World()
# world.get_physics_context().set_gravity(0.0)
# world.set_gravity(value=-9.81 / meters_per_unit())

rclpy.init()
node = rclpy.create_node("isaac_picking_simulator")
publisher = node.create_publisher(Image, "/camera/camera/color/image_raw", 1)
bin_state_publisher = node.create_publisher(MarkerArray, "/bin_state", 1)
br = cv_bridge.CvBridge()


def publish_image(img):
    def crop_center_and_resize(img):
        width, height = [768, 540]
        cam_height = img.shape[0]
        cam_width = img.shape[1]
        output_img_size = [512, 360]
        cropped_img = img[
            int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
            int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
        ]
        return cv2.resize(cropped_img, output_img_size)

    img = crop_center_and_resize(img)
    msg = br.cv2_to_imgmsg(img, encoding="rgb8")
    publisher.publish(msg)


message_id = 0
object_info = ObjectInfo("ycb_conveni_v1")


def publish_bin_state(task):
    global message_id
    bin_state = []
    for product in task.get_active_products():
        p, o = product.get_world_pose()
        o[0], o[1], o[2], o[3] = o[1], o[2], o[3], o[0]
        bin_state.append((product.name, (p, o)))

    # markerD = Marker()
    # markerD.header.frame_id = 'fmap_frame'
    # markerD.action = markerD.DELETEALL
    marker_array = MarkerArray()
    # marker_array.markers.append(markerD)

    rgba = [0.5, 0.5, 0.5, 0.4]
    scale = [1.0, 1.0, 1.0]

    for name, (xyz, quat) in bin_state:
        marker = Marker()
        marker.type = Marker.MESH_RESOURCE
        marker.header.frame_id = "fmap_frame"
        marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        # marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        marker.lifetime = rclpy.duration.Duration().to_msg()
        marker.id = message_id
        marker.action = marker.ADD
        message_id += 1

        mesh_file, _ = object_info.rviz_mesh_file(name)
        marker.mesh_resource = mesh_file
        marker.mesh_use_embedded_materials = True
        xyz = xyz.tolist()
        quat = quat.tolist()
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        marker_array.markers.append(marker)
    bin_state_publisher.publish(marker_array)


from isaacsim.core.utils.transformations import pose_from_tf_matrix, tf_matrix_from_pose
import transforms3d as tf
import json
import numpy as np

def load_grasp_data():
    grasp_data_file = '/home/ryo/Program/isaac_sim_grasping/graspit_grasps/franka_panda/franka_panda-052_extra_large_clamp.json'
    with open(grasp_data_file) as f:
        data = json.load(f)
    poses = np.asarray(data['pose'])  # shape = (9830, 7)
    dofs = np.asarray(data['final_dofs'])
    indices = np.argsort(data['fall_time'])[::-1]
    poses_and_dofs = poses[indices], dofs[indices]
    return poses_and_dofs


Tycb_mggs = {  # given from Meshlab (ICP)
    '052_extra_large_clamp': np.array([
        [ 9.99999514e-01, -1.05736320e-03,  6.06697628e-04, -2.02874644e-02],
        [ 1.05417675e-03,  9.99985434e-01,  5.23968244e-03, -3.58058714e-02],
        [-6.12228836e-04, -5.23904553e-03,  9.99986177e-01, 1.72968221e-02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
}

def gripper_pose_in_world(task, pose, name='052_extra_large_clamp'):
    for product in task._env.products:
        if product.name == name:
            obj_pos, obj_ori = product.get_world_pose()
    Twld_ycb = tf_matrix_from_pose(translation=obj_pos, orientation=obj_ori)
    Tmgg_palm = tf_matrix_from_pose(translation=pose[:3], orientation=pose[3:])
    # Tycb_mgg = np.array([  # given from Meshlab (ICP)
    #     [ 1.00000000e+00,  5.06082872e-20, -6.05749121e-20, 0.00000000e+00],
    #     [ 2.71320677e-19,  1.00000000e+00, -8.08456633e-19, 0.00000000e+00],
    #     [ 1.01498696e-20, -4.13673160e-19,  1.00000000e+00, 0.00000000e+00],
    #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    Tycb_mgg = Tycb_mggs[name]

    # Tpalm_ur = tf_matrix_from_pose(
    #     translation=[0., 0., -0.115],
    #     orientation=tf.euler.euler2quat(0., 0., 0., axes='sxyz')
    # )  # franka canonical frame -> UR5e tool0 frame

    Tpalm_franka = tf_matrix_from_pose(
        translation=[0., 0., 0.05],
        orientation=tf.euler.euler2quat(0., 0., 0., axes='sxyz')
    )  # franka canonical frame -> franka tool_center frame

    # print(f'Twld_ycb={Twld_ycb}')
    # print(f'Tmgg_palm={Tmgg_palm}')
    # pose_w = pose_from_tf_matrix(Twld_ycb @ Tycb_mgg @ Tmgg_palm @ Tpalm_ur)

    pose_w = pose_from_tf_matrix(Twld_ycb @ Tycb_mgg @ Tmgg_palm @ Tpalm_franka)
    return pose_w


## Grasp sampler (MIMO)
import trimesh
from mimo.data_gen.panda_sample import PandaGripper, sample_multiple_grasps
from dataset.object_loader import ObjectInfo

oi = ObjectInfo()

def sample_grasps_from_mesh(object_name, n_grasps=20):
    gripper = PandaGripper()
    obj_mesh = trimesh.load(oi.obj_file(object_name, with_scale=False), force='mesh')
    trans, quality = sample_multiple_grasps(n_grasps, obj_mesh, gripper, systematic_sampling=False)
    quality = np.array(quality["quality_antipodal"])
    trans = trans[quality > 0.06]
    grasp = trans.tolist()
    print(f"Number of high quality grasps: {len(grasp)}")
    return grasp


# scene_idx = 13
# lifting_targets = ['052_extra_large_clamp', 'jif', '004_sugar_box', 'java_curry_chukara', 'kinokonoyama', 'oi_ocha_350ml']

lifting_problems = [
    (2, 'calorie_mate_cheese'),    
    # (8, 'calorie_mate_cheese'), 
    # (14, '052_extra_large_clamp', 0.07),    
    # (22, '010_potted_meat_can'),
    # (27, 'chip_star', 0.10),
    # (29, '043_phillips_screwdriver', 0.05),
    # (35, 'calorie_mate_cheese'),
    # (44, '008_pudding_box'),
    # (45, '024_bowl', 0.07),    
    # (47, 'kinokonoyama'),         
    # (65, 'kinokonoyama'),
    # (86, '033_spatula', 0.07),        
    # (100, 'gogotea_straight', 0.10),          
    # (121, '052_extra_large_clamp', 0.07),
    # (69, 'pan_meat'), 
    # (69, 'chocoball'),
    ]

# scene_idx, lifting_target = lifting_problems[-1]

def get_product_world_pose(task, target_name: str):
    for product in task._env.products:
        if product.name == target_name:
            return product.get_world_pose()  # get_world_pose() returns scalar-first quaternion


def gripper_pose_in_world_mimo(task, grasp, target_name: str, pregrasp_opening=0.13):
    """
    The depth of grasp is dependent on the pregrasp opening.
    """

    def pos_euler2mat(pos, euler):
        return tf.affines.compose(pos, tf.euler.euler2mat(euler[0], euler[1], euler[2], axes='sxyz'), np.ones(3))
    def pos_quat2mat(pos, quat):
        return tf.affines.compose(pos, tf.quaternions.quat2mat(quat), np.ones(3))

    obj_pos, obj_quat = get_product_world_pose(task, target_name)
    Tworld_obj = pos_quat2mat(obj_pos, obj_quat)
    # Tmimo_franka = pos_euler2mat([0, 0, 0.09], [0, 0, np.pi/2])

    ## 2f-140 grasp depth
    grasp_depth = -0.168 * pregrasp_opening - 0.10716
    Tmimo_gripper = pos_euler2mat([0, 0, grasp_depth], [0, 0, np.pi/2])
    return pose_from_tf_matrix(Tworld_obj @ grasp @ Tmimo_gripper)  # pose_from_tf_matrix() returns scalar-first quaternion

HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()

my_world: World = World(stage_units_in_meters=1.0)

task = TabletopPickingTask(static_path=HELLO_ISAAC_ROOT / "aist_sb_ur5e/static", robot="ur5e")
my_world.add_task(task=task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(name=task.get_params()["robot_names"]["value"][0])
target: XFormPrim = my_world.scene.get_object(name=task.get_params()["x_form_prim_names"]["value"][0])
gripper: ParallelGripper = task.get_params()["gripper"]["value"]

print(f'PD GAIN: {ur5e._articulation_controller.get_gains()}')
print(f'MAX EFFORTS: {ur5e._articulation_controller.get_max_efforts()}')
# ur5e._articulation_controller.set_max_efforts([1e+5]*7 + [30.0, 0.0])
stiffnesses = [1500.0]*6 + [0, 5e+4, 0, 5e+4, 0, 0]
dampings = [1500.0]*6 + [0, 4e+4, 0, 4e+4, 0, 0]
# ur5e._articulation_controller.set_gains(kps=stiffnesses, kds=dampings)
# print(f'PD GAIN (2): {ur5e._articulation_controller.get_gains()}')
# print(f'MAX EFFORTS (2): {ur5e._articulation_controller.get_max_efforts()}')


rmpflow_controller = RMPFlowController(
    # name="cspace_controller",
    robot_articulation=ur5e,
    robot_description_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/robot_descriptor.yml"),
    rmpflow_config_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/ur5e_rmpflow_common.yml"),
    urdf_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/urdf/ur5e.urdf"),
)

ik_solver = KinematicsSolver(
    robot_articulation=ur5e,
)


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


ik_solver = KinematicsSolver(robot_articulation=ur5e)
ik_solver._kinematics.set_robot_base_pose(*ur5e.get_world_pose())


def reset_robot_state():
    js = ur5e.get_joints_default_state()
    ur5e.set_joint_positions(js.positions)


class Recorder:
    def __init__(self, 
                 task, 
                 crop_size=[768,540],
                 output_image_size=[320,240]):

        self._task = task
        self._data_dir = Path("picking_experiment_results")
        self._crop_size = crop_size
        self._output_image_size = output_image_size

    def new_episode(self, name):
        self._frameNo = 0
        self._episode_name = name
        p = self._data_dir / self._episode_name
        p.mkdir(parents=True, exist_ok=True)

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

        for o in task.get_active_products():
            cs = task._env._product_sensors[o.name]
            current_frame = cs.get_current_frame()
            if current_frame['in_contact']:
                for contact in current_frame['contacts']:
                    objectA = contact["body0"]
                    objectB = contact["body1"]
                    contact_positions.append(contact["position"])
                    contact_normals.append(contact["normal"])
                    impulse_values.append(np.linalg.norm(contact["impulse"]))
                    contacting_objects.append((objectA, objectB))
                    # print(f' {objectA}, {objectB}, {contact["impulse"]}')

        # remove duplicated contacts
        # contact_positions, uidx = np.unique(contact_positions, axis=0, return_index=True)
        # contact_normals = [contact_normals[idx] for idx in uidx]
        # impulse_values = [impulse_values[idx] for idx in uidx]
        # contacting_objects = [contacting_objects[idx] for idx in uidx]

        pd.to_pickle(
            (contact_positions, impulse_values, contacting_objects, contact_normals),
            self._data_dir / self._episode_name / f"contact_raw_data{self._frameNo:05d}.pkl"
        )

    def save_image(self):
        def crop_center_and_resize(img):
            width, height = self._crop_size
            cam_height = img.shape[0]
            cam_width = img.shape[1]
            cropped_img = img[
                int((cam_height - height) / 2) : int((cam_height - height) / 2 + height),
                int((cam_width - width) / 2) : int((cam_width - width) / 2 + width),
            ]
            return cv2.resize(cropped_img, self._output_image_size)

        rgb = self._task._cameras[0].get_rgb()  # top camera
        rgb_path = self._data_dir / self._episode_name / f"rgb{self._frameNo:05}.jpg"
        output_rgb = crop_center_and_resize(rgb)
        cv2.imwrite(str(rgb_path), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

    def save(self):
        self.save_robot_state()
        self.save_product_state()
        self.save_contact_state()
        self.save_image()
        self._frameNo += 1


def do_lifting(recorder, end_of_episode = 300, end_of_grasping = 200):
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
                    pos0, ori0 = ur5e._kinematics.compute_forward_kinematics('tool0', ur5e.get_joint_positions()[:6])
                
            elif end_of_grasping < counter < end_of_episode:  # do lifting
                target_position, target_orientation = ur5e._kinematics.compute_forward_kinematics('tool0', ur5e.get_joint_positions()[:6])
                target_position[2] += 0.02

                # target_orientation = tf.quaternions.mat2quat(target_orientation)
                target_orientation = tf.quaternions.mat2quat(ori0)                

                next_position, next_orientation = target_controller.forward(
                    position=target_position,
                    orientation=target_orientation,
                )
                action: ArticulationAction = rmpflow_controller.forward(
                    target_end_effector_position=next_position,
                    target_end_effector_orientation=next_orientation,
                )
                ur5e.get_articulation_controller().apply_action(control_actions=action)
                recorder.save()

            counter += 1


from collections import deque

class GraspSampler:
    def __init__(self):
        self._grasp_queue = deque()

    def sample_grasp(self, object_name: str):
        try:
            grasp = self._grasp_queue.popleft()
        except IndexError:
            self._grasp_queue.extend(sample_grasps_from_mesh(object_name, n_grasps=200))
            grasp = self._grasp_queue.popleft()

        return grasp

grasp_sampler = GraspSampler()


def sample_pregrasp_opening():
    return np.random.choice([0.06])


def find_feasible_grasp(scene_idx, lifting_target):
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

                    pregrasp_opening = sample_pregrasp_opening()
                    target_gripper_joint_position = convert_to_joint_angle(pregrasp_opening)
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
                return grasp, pregrasp_opening

            counter += 1


def collect_successful_grasps():
    for problem in lifting_problems[-1:]:
        counter = 0
        successful_grasps = []

        print(f'PROBLEM: {problem}')
        scene_idx, lifting_target = problem

        while True:
            if len(successful_grasps) >= 3:
                print(f'PROBLEM SOLVED: {(problem, successful_grasps)}')
                break

            grasp, pregrasp_opening = find_feasible_grasp(*problem)  # or set_feasible_grasp()
            do_lifting()

            if get_product_world_pose(task, lifting_target)[0][2] > 0.85:
                successful_grasps.append((pregrasp_opening, grasp))
                print(f'# of successful grasps = {len(successful_grasps)}')


from generated_grasps import episodes

def run_successful_grasps(lifting_method='UP'):
    recorder = Recorder(task)

    for problem, successful_grasps in episodes:
        counter = 0

        print(f'PROBLEM: {problem}')
        scene_idx, lifting_target = problem
        recorder.new_episode(name=f'{scene_idx}__{lifting_target}__{lifting_method}')

        for pregrasp_opening, grasp in successful_grasps:
            reset_robot_state()
            task.load_bin_state(scene_idx)
            g_pos, g_ori = gripper_pose_in_world_mimo(task, grasp, lifting_target)
            target.set_world_pose(position=g_pos, orientation=g_ori)  # set_world_pose() takes scalar-first quaternion

            position_tolerance = 0.002
            orientation_tolerance = 0.02
            ik_action, success = ik_solver.compute_inverse_kinematics(g_pos, g_ori, position_tolerance, orientation_tolerance)

            target_gripper_joint_position = convert_to_joint_angle(pregrasp_opening)
            target_joint_positions = set_joint_positions_UR5e(ik_action.joint_positions , target_gripper_joint_position) 
            ur5e.set_joint_velocities(np.zeros(len(target_joint_positions)))
            
            do_lifting(recorder)




# collect_successful_grasps()
run_successful_grasps()


simulation_app.close()
node.destroy_node()
rclpy.shutdown()
