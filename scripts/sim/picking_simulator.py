from operator import itemgetter
from pathlib import Path

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False, "multi_gpu": False})
import cv2
import cv_bridge
import rclpy
from aist_sb_ur5e.controller import RMPFlowController
# from aist_sb_ur5e_controller import SpaceMouseController
from aist_sb_ur5e_controller import KeyboardController
from dataset.object_loader import ObjectInfo
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper

# from std_msgs.msg import String
from sensor_msgs.msg import Image

# from aist_sb_ur5e.task import ConveniPickupTask
from sim.task.tabletop_picking_task import TabletopPickingTask
from visualization_msgs.msg import Marker, MarkerArray

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
    dofs = np.asarray(data['dofs'])
    indices = np.argsort(data['fall_time'])[::-1]
    poses = poses[indices]
    return poses

def gripper_pose_in_world(task, pose, name='052_extra_large_clamp'):
    for product in task._convencience_store.products:
        if product.name == name:
            obj_pos, obj_ori = product.get_world_pose()
    Twld_ycb = tf_matrix_from_pose(translation=obj_pos, orientation=obj_ori)
    Tmgg_palm = tf_matrix_from_pose(translation=pose[:3], orientation=pose[3:])
    Tycb_mgg = np.array([  # given from Meshlab (ICP)
        [ 1.00000000e+00,  5.06082872e-20, -6.05749121e-20, 0.00000000e+00],
        [ 2.71320677e-19,  1.00000000e+00, -8.08456633e-19, 0.00000000e+00],
        [ 1.01498696e-20, -4.13673160e-19,  1.00000000e+00, 0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    Tpalm_ur = tf_matrix_from_pose(
        translation=[0., 0., 0.],
        orientation=tf.euler.euler2quat(np.pi/2., 0., 0., axes='sxyz')
    )  # franka canonical frame -> UR5e tool0 frame

    # print(f'Twld_ycb={Twld_ycb}')
    # print(f'Tmgg_palm={Tmgg_palm}')
    pose_w = pose_from_tf_matrix(Twld_ycb @ Tycb_mgg @ Tmgg_palm @ Tpalm_ur)
    return pose_w


HELLO_ISAAC_ROOT = Path("~/Program/hello-isaac-sim").expanduser()

my_world: World = World(stage_units_in_meters=1.0)

task = TabletopPickingTask(static_path=HELLO_ISAAC_ROOT / "aist_sb_ur5e/static")
my_world.add_task(task=task)
my_world.reset()

ur5e: Articulation = my_world.scene.get_object(name=task.get_params()["robot_names"]["value"][0])
target: XFormPrim = my_world.scene.get_object(name=task.get_params()["x_form_prim_names"]["value"][0])
gripper: ParallelGripper = task.get_params()["gripper"]["value"]

rmpflow_controller = RMPFlowController(
    robot_articulation=ur5e,
    robot_description_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/robot_descriptor.yml"),
    rmpflow_config_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/rmpflow/ur5e_rmpflow_common.yml"),
    urdf_path=str(HELLO_ISAAC_ROOT / "aist_sb_ur5e/static/urdf/ur5e.urdf"),
)
# target_controller = SpaceMouseController(device_type="SpaceMouse Compact", rotate_gain=0.3)
target_controller = KeyboardController()

scene_idx = 2
# task.load_bin_state(scene_idx)

poses = load_grasp_data()
frame_no = 0

while simulation_app.is_running():
    frame_no += 1
    my_world.step(render=True)
    if my_world.is_playing():

        task.load_bin_state(scene_idx)

        target_position, target_orientation = target.get_world_pose()
        next_position, next_orientation = target_controller.forward(
            position=target_position,
            orientation=target_orientation,
        )

        pose = poses[int(frame_no / 5)]
        g_pos, g_ori = gripper_pose_in_world(task, pose)
        target.set_world_pose(position=g_pos, orientation=g_ori)

        # target.set_world_pose(
        #     position=next_position,
        #     orientation=next_orientation,
        # )

        action: ArticulationAction = rmpflow_controller.forward(
            target_end_effector_position=next_position,
            target_end_effector_orientation=next_orientation,
        )
        # ur5e.get_articulation_controller().apply_action(control_actions=action)

        optional_action: str | None = target_controller.get_gripper_action()
        if optional_action is not None:
            gripper_action: ArticulationAction = gripper.forward(optional_action)
            gripper.apply_action(ArticulationAction(joint_positions=itemgetter(7, 9)(gripper_action.joint_positions)))

        img = task._cameras[0].get_rgb()
        if len(img.shape) == 3:
            publish_image(img)
        publish_bin_state(task)


simulation_app.close()

node.destroy_node()
rclpy.shutdown()
