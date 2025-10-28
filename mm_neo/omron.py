"""
@file: omron.py

Simple class to interface with omron LD-60 and send a target goal
"""

import spatialmath as sm
from spatialmath_rospy import to_ros

import rospy
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from actionlib import SimpleActionClient


class Omron:
    def __init__(self):
        self.client = SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base server")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base server")
        self.goal = MoveBaseGoal()

    def send_goal(self, pose: sm.SE3):
        self.goal.target_pose.header.frame_id = "map"
        self.goal.target_pose.pose = to_ros(pose)
        self.client.send_goal(self.goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            rospy.loginfo("Goal sent")
            return self.client.get_result()
        return self.client.get_result()
