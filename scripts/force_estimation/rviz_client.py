# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3


class RVizClient:
    def __init__(self):
        self._message_id = 0
        self._node_name = "force_distribution_publisher"
        self.start_ros_node()

    def start_ros_node(self):
        rospy.init_node(self._node_name)
        self._pub = rospy.Publisher("scene_objects", MarkerArray, queue_size=1)
        self.rate = rospy.Rate(30)
        self._base_frame_id = "map"

    def show(self):
        self._pub.publish(self._markerArray)

    def delete_all(self):
        self._message_id = 0
        markerD = Marker()
        markerD.header.frame_id = self._base_frame_id
        markerD.action = markerD.DELETEALL
        self._markerArray = MarkerArray()
        self._markerArray.markers.append(markerD)

    def draw_mesh(self, mesh_file, pose, rgba):
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
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

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

        for position, rgba, in zip(points, rgbas):
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

    def draw_arrows(self, tails, tips, rgba=[0.2, 0.5, 1.0, 0.3], scale=[0.0015, 0.003, 0.002]):
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
