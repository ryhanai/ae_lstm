import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import colorsys

rospy.init_node("force_distribution_publisher")
pub = rospy.Publisher("scene_objects", MarkerArray, queue_size=1)
rate = rospy.Rate(30)

def mesh_message(mesh_file, message_id, pose, rgba=(0.5,0.5,0.5,0.3)):
    marker = Marker()
    marker.type = marker.MESH_RESOURCE    
    marker.mesh_resource = mesh_file
    marker.mesh_use_embedded_materials = True
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.id = message_id
    marker.action = marker.ADD
    marker.lifetime = rospy.Duration()
    
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
    return marker
    
def publish_bin_state(bin_state, force_distribution, draw_bin=True):
    markerD = Marker()
    markerD.header.frame_id = "map"
    markerD.action = markerD.DELETEALL

    mid = 0
    markerArray = MarkerArray()
    markerArray.markers.append(markerD)

    if draw_bin:
        marker = mesh_message("package://fmap_visualizer/meshes_extra/sony_box.dae",
                              mid,
                              ((0,0,0.79),(0,0,0,1)),
                              (0.5,0.5,0.5,0.2))
        markerArray.markers.append(marker)
        mid += 1
    
    for object_state in bin_state:
        name, pose = object_state
        marker = mesh_message("package://fmap_visualizer/meshes/{}/google_16k/textured.dae".format(name),
                              mid,
                              pose,
                              (0.5,0.5,0.5,0.3))
        markerArray.markers.append(marker)
        mid += 1

    if force_distribution != None:
        marker = Marker()
        marker.type = marker.POINTS

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.id = mid
        marker.action = marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.0015
        marker.scale.y = 0.0015
        marker.scale.z = 0.0015

        positions, fvals = force_distribution
        fmax = np.max(fvals)
        fmin = np.min(fvals)
        if fmax - fmin > 1e-3:
            #std_fvals = np.clip(5.0 * (fvals - fmin) / (fmax - fmin), 0.0, 1.0)
            std_fvals = (fvals - fmin) / (fmax - fmin)
            for (x,y,z), f in zip(positions, std_fvals):
                if f > 0.08:
                    marker.points.append(Point(x,y,z))
                    r,g,b = colorsys.hsv_to_rgb(1./3 * (1-f),1,1)
                    marker.colors.append(ColorRGBA(r, g, b, 1))

        marker.lifetime = rospy.Duration()
        markerArray.markers.append(marker)
        
    pub.publish(markerArray)

    
# while not rospy.is_shutdown():
#     publish_bin_state(bin_state, force_density)
#     rate.sleep()
