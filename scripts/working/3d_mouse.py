import rospy
from geometry_msgs.msg import Vector3


last_msg = None

def callback(msg):
    global last_msg
    last_msg = msg
    # rospy.loginfo("%s", msg)

def setReceiver():
    rospy.init_node("spacenav_receiver")
    rospy.Subscriber("/spacenav/offset", Vector3, callback)

