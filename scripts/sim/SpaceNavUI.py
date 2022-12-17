from SIM_KITTING import *

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Joy

UI_EV_LEFT_CLICK = 0
UI_EV_RIGHT_CLICK = 1

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

    def getControlSignal(self):
        l = self.last_msg.linear
        a = self.last_msg.angular
        v = 0.03
        w = 0.05
        return [v*l.x, v*l.y, v*l.z], [w*a.x, w*a.y, w*a.z]

    def getEventSignal(self):
        if self.last_joy_msg.buttons[0] == 1: # left button is down, task_completed
            return UI_EV_LEFT_CLICK
        if self.last_joy_msg.buttons[1] == 1:
            return UI_EV_RIGHT_CLICK

            #print("L: ", control.last_msg.linear)
            #print("A: ", control.last_msg.angular)
            #env.moveEF([v * (-control.last_msg.y), v * control.last_msg.x, v * control.last_msg.z])

    def spacenav_callback(self, msg):
        self.last_msg = msg
        # rospy.loginfo("%s", msg)

    def joy_callback(self, msg):
        self.last_joy_msg = msg
