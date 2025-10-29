import copy
import sys
import numpy as np
from spatialmath_rospy import to_spatialmath, to_ros
import swift
import spatialgeometry as sg
import spatialmath as sm
import roboticstoolbox as rtb

# Ros stuff
import rospy
from cv_bridge import CvBridge
import message_filters
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState, CameraInfo

# custom libraries
from mm_neo.scanner import Scanner_vertical
from neural_robot import neural_frankie as nf
from mm_neo.mm_controllers import mm_neo, VelocityController
from nerf_tools.dataset.nerf_dataset import RosToNeRF
from remomac.msg import MobileManipulatorVelocity


class RosController():
    def __init__(self, initial_pose: sm.SE3, env: swift.Swift):
        self.robot = nf.NeuralFrankie("points_2")
        
        self.nerf_pose_msg = PoseStamped()
        self.nerf_pose_msg.header.frame_id = "map"
        
        self.optical_frame_msg = PoseStamped()
        self.optical_frame_msg.header.frame_id = "map"
        
        self.odom_sub = rospy.Subscriber("/frankie/odom", Odometry, self.callback_odom)
        self.joint_state_sub = rospy.Subscriber("/frankie/joint_states", JointState, self.callback_joint_state) 
        self.camera_pose_sub = rospy.Subscriber("/camera_pose", PoseStamped, self.callback_camera_pose)
        
        self.mm_pub = rospy.Publisher("/frankie/remomac/mm_cmd_vel", MobileManipulatorVelocity, queue_size=10)
        self.nerf_pose_pub = rospy.Publisher("/frankie/remomac/nerf_pose", PoseStamped, queue_size=10)
        self.optical_frame_pub = rospy.Publisher("/frankie/remomac/optical_frame", PoseStamped, queue_size=10)
        env.add(self.robot)
        self.vel_msg = MobileManipulatorVelocity() 
        self.vel_msg.ee_velocity_ref_frame = self.vel_msg.END_EFFECTOR
        
        
        
        
        self.controller_rate = rospy.Rate(50) # 30 Hz
        self.scanning_index = 0
        self.scanning_sequence = Scanner_vertical(0.3, 0.1, 4,3, initial_pose, x_offset=0.75, z_offset= 1.15)
        self.bridge = CvBridge()
        self.__add_debug_poses(env, self.scanning_sequence.T_WS_poses)
        env.set_camera_pose(initial_pose.t + [4.0,0,4.0], initial_pose.t)
        env.step()
        camera_info = rospy.wait_for_message("/camera_info", CameraInfo)

        self.dataset = RosToNeRF("results/unity_scans", camera_info)
        self.controller = mm_neo(self.robot)
        self.vel_controller = VelocityController(self.robot)
        self.env = env
        
    def step_controller(self):
        if self.scanning_index >= len(self.scanning_sequence.T_WS_poses):
            self.dataset.save("transforms.json")
            sys.exit(1)
        
        target_pose = self.scanning_sequence.get_pose(self.scanning_index)
        T_We = self.robot.fkine(self.robot._q)
       
        T_eEp = np.linalg.inv(target_pose) @ T_We.A

        # spatial error
        et = np.sum(np.abs(T_eEp[0:3,3]))
        
        if et < 0.01:
            self.scanning_index += 1
            img = rospy.wait_for_message("/cam", Image)
            depth_image = rospy.wait_for_message("/depth_cam", Image)
            cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            camera_pose = to_spatialmath(rospy.wait_for_message("/camera_pose", PoseStamped).pose) * sm.SE3.Rx(np.pi)
            self.dataset.record_frame(cv_image, cv_depth_image, camera_pose)
            return
       
        
        v, _ = rtb.p_servo(T_We, target_pose, 1.5)

        self.vel_msg.ee_velocity.linear.x = v[0]
        self.vel_msg.ee_velocity.linear.y = v[1]
        self.vel_msg.ee_velocity.linear.z = v[2]
        
        self.vel_msg.ee_velocity.angular.x = v[3]
        self.vel_msg.ee_velocity.angular.y = v[4]
        self.vel_msg.ee_velocity.angular.z = v[5]
        
        
        self.mm_pub.publish(self.vel_msg)
        env.step()
        return


    def callback_odom(self, odom: Odometry):
        # rospy.loginfo("callback_odom")
        base_new = to_spatialmath(odom.pose.pose)
        self.robot._T = base_new.A
        return

    def callback_camera_pose(self, camera_pose: PoseStamped):
        # rospy.loginfo("callback_odom")
        pose = to_spatialmath(camera_pose.pose) * sm.SE3.Rx(np.pi/2) * sm.SE3.Rz(np.pi)
        
        self.nerf_pose_msg.pose = to_ros(pose)
        self.nerf_pose_pub.publish(self.nerf_pose_msg)

        optical_pose = copy.deepcopy(pose) * sm.SE3.Rx(np.pi)
        self.optical_frame_msg.header.stamp = camera_pose.header.stamp
        self.optical_frame_msg.pose = to_ros(optical_pose)
        self.optical_frame_pub.publish(self.optical_frame_msg)


        return




    def callback_joint_state(self, joint: JointState):
        rospy.loginfo("callback_joint_state")
        self.robot._q[2:] = np.array(joint.position[:-1])
        self.robot.grippers[0]._q = np.array([joint.position[-1], joint.position[-1]])
        return

    def __add_debug_poses(self, env, poses):
        for i in poses:
            axis = sg.Axes(0.1)
            axis.T = i
            env.add(axis)
            env.step()
        return


index = -1

rospy.init_node("frankie_scanner")

env = swift.Swift()
env.launch(realtime= True, headless= True)
rospy.sleep(2)



# robot._T = base_new.A


print("Starting scan")

initial_pose = to_spatialmath(rospy.wait_for_message("/frankie/odom", Odometry).pose.pose)

Control = RosController(initial_pose, env)

while not rospy.is_shutdown():
    Control.step_controller()
    
