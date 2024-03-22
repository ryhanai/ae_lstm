# -*- coding: utf-8 -*-

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, MarkerArray


def normalizeQuaternion(quaternion_msg):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm ** (-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s


class RVizClient:
    def __init__(self):
        self._message_id = 0
        self._node_name = "force_distribution_publisher"
        self._base_frame_id = "map"
        self._6dof = True
        self._interactive_marker_size = 0.08
        self.start_ros_node()

    def start_ros_node(self):
        rospy.init_node(self._node_name)
        self._pub = rospy.Publisher("scene_objects", MarkerArray, queue_size=1)

        self._interactive_marker_server = InteractiveMarkerServer("simple_marker")
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._base_frame_id
        int_marker.name = "object_center"
        int_marker.description = "Object Center"
        int_marker.pose.position = Point(0.0, 0.0, 0.80)
        int_marker.scale = self._interactive_marker_size
        box_marker = self._make_marker(Marker.SPHERE)
        box_marker.pose = int_marker.pose
        box_marker.scale = Vector3(0.005, 0.005, 0.005)
        box_marker.color = ColorRGBA(1, 0, 0, 0.5)

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(box_marker)
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = "move_x"
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = "move_y"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = "move_z"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        if self._6dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        self._interactive_marker_server.insert(int_marker, self.processInteractiveMarkerFeedback)
        self._interactive_marker_server.applyChanges()
        self._interactiveMarkerCallback = None

        self._interactive_marker_pose = (np.array([0, 0, 0.78]), np.array([0, 0, 0, 1]))
        self.rate = rospy.Rate(30)

    def processInteractiveMarkerFeedback(self, feedback):
        #  print(feedback.pose.position.x, ', ', feedback.pose.position.y, ', ', feedback.pose.position.z)
        p = feedback.pose.position
        q = feedback.pose.orientation
        self._interactive_marker_pose = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))
        if self._interactiveMarkerCallback != None:
            self._interactiveMarkerCallback()

    def getObjectPosition(self):
        return self._interactive_marker_pose[0]

    def getInteractiveMarkerPose(self):
        return self._interactive_marker_pose

    def addInteractiveMarkerCallback(self, func):
        self._interactiveMarkerCallback = func

    def show(self):
        self._pub.publish(self._markerArray)

    def delete_all(self):
        self._message_id = 0
        markerD = Marker()
        markerD.header.frame_id = self._base_frame_id
        markerD.action = markerD.DELETEALL
        self._markerArray = MarkerArray()
        self._markerArray.markers.append(markerD)

    def draw_mesh(self, mesh_file, pose, rgba, scale=[1.0, 1.0, 1.0]):
        marker = self._make_marker(Marker.MESH_RESOURCE)
        marker.mesh_resource = mesh_file
        marker.mesh_use_embedded_materials = True

        xyz, quat = pose
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

        self._markerArray.markers.append(marker)

    def draw_points(self, points, rgbas, point_size=0.0015):
        marker = self._make_marker(Marker.POINTS)

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = point_size
        marker.scale.y = point_size
        marker.scale.z = point_size

        for (
            position,
            rgba,
        ) in zip(points, rgbas):
            marker.points.append(Point(*position))
            marker.colors.append(ColorRGBA(*rgba))

        self._markerArray.markers.append(marker)

    def _make_marker(self, marker_type):
        marker = Marker()
        marker.type = marker_type
        marker.header.frame_id = self._base_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration()
        marker.id = self._message_id
        marker.action = marker.ADD
        self._message_id += 1
        return marker

    def draw_arrow(self, tail, tip, rgba, scale):
        marker = self._make_marker(Marker.ARROW)
        marker.pose.orientation.y = 0
        marker.pose.orientation.w = 1
        marker.scale = Vector3(*scale)
        marker.color = ColorRGBA(*rgba)
        marker.points = [Vector3(*tail), Vector3(*tip)]

        self._markerArray.markers.append(marker)

    def draw_arrows(self, tails, tips, rgba=[0.2, 0.5, 1.0, 0.3], scale=[0.001, 0.001, 0.000]):
        """
        This methods is very slow because it sends a topic for each arrow.
        """
        for tail, tip in zip(tails, tips):
            self.draw_arrow(tail, tip, rgba, scale)

    def draw_sphere(self, center, rgba, scale):
        marker = self._make_marker(Marker.SPHERE)
        marker.pose.position = Point(*center)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = Point(*scale)
        marker.color = ColorRGBA(*rgba)
        self._markerArray.markers.append(marker)
        marker.scale = Point(*scale)
        marker.color = ColorRGBA(*rgba)
        self._markerArray.markers.append(marker)
        marker.scale = Point(*scale)
        marker.color = ColorRGBA(*rgba)
        self._markerArray.markers.append(marker)
        marker.scale = Point(*scale)
        marker.color = ColorRGBA(*rgba)
        self._markerArray.markers.append(marker)
        marker.scale = Point(*scale)
        marker.color = ColorRGBA(*rgba)
        self._markerArray.markers.append(marker)
