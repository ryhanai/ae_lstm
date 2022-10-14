import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import colorsys

rospy.init_node("force_distribution_publisher")
pub = rospy.Publisher("scene_objects", MarkerArray, queue_size=1)
rate = rospy.Rate(25)

objects = [
  '004_sugar_box',
  '006_mustard_bottle',
  '007_tuna_fish_can',
  '010_potted_meat_can',
  '026_sponge', # soft body
  '011_banana',
  '012_strawberry',
  '013_apple',
  '016_pear',
]

bin_state = [
('006_mustard_bottle',
 ((0.009855880221905401, 0.028207773275998566, 0.9440576706538131),
  (0.011163616068588768,
   0.4002768767076521,
   -0.7339873110271562,
   0.5485584954228357))),
 ('026_sponge',
 ((0.0012394144997380288, -0.050092795870262746, 0.9854077186320029),
  (0.71195269789557,
   -0.16976341416280127,
   -0.14132606874538753,
   0.6665813389707085))),
 ('010_potted_meat_can',
 ((-0.046273377364551976, 0.00846329317874374, 0.9703276334581414),
  (-0.08147422565819812,
   0.3094228301753092,
   -0.7295036407507537,
   0.6045195620086203)))
 ]

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
        marker = mesh_message("package://ycb_dataset/meshes_extra/sony_box.dae",
                              mid,
                              ((0,0,0.79),(0,0,0,1)),
                              (0.5,0.5,0.5,0.2))
        markerArray.markers.append(marker)
        mid += 1
    
    for object_state in bin_state:
        name, pose = object_state
        print(name)
        marker = mesh_message("package://ycb_dataset/meshes/{}/google_16k/textured.dae".format(name),
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

        marker.scale.x = 0.002
        marker.scale.y = 0.002
        marker.scale.z = 0.002

        positions, fvals = force_distribution
        for (x,y,z), f in zip(positions, fvals):
            v = min(f/500, 1.0)
            if (v > 0.1):
                marker.points.append(Point(x,y,z))
                r,g,b = colorsys.hsv_to_rgb(1./3 * (1-v),1,1)
                marker.colors.append(ColorRGBA(r, g, b, 1))

        marker.lifetime = rospy.Duration()
        markerArray.markers.append(marker)
        
    pub.publish(markerArray)

    
# while not rospy.is_shutdown():
#     publish_bin_state(bin_state, force_density)
#     rate.sleep()
