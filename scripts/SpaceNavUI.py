from SIM_KITTING import *
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Joy

class SpaceNavUI(UI):
    def __init__(self):
        super().__init__()
        self.last_msg = Twist()
        self.last_joy_msg = Joy()
        rospy.init_node("spacenav_receiver")
        rospy.Subscriber("/spacenav/twist", Twist, self.spacenav_callback, queue_size=1)
        rospy.Subscriber("/spacenav/joy", Joy, self.joy_callback, queue_size=1)

    def update(self):
        pass
            
    def spacenav_callback(self, msg):
        self.last_msg = msg
        rospy.loginfo("%s", msg)

    def joy_callback(self, msg):
        self.last_joy_msg = msg