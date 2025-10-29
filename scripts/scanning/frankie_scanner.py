import sys
import numpy as np
from spatialmath_rospy import to_ros, to_spatialmath
import swift
import spatialgeometry as sg
import spatialmath as sm
import spatialmath.base as smb
from typing import List

# Ros stuff
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from sensor_msgs.msg import Image, JointState, CameraInfo
from mm_neo.configs.controllers import NeoConfig
from mm_neo.configs.sdf import VSConfig
import tf

# custom libraries
from mm_neo.scanner import Scanner_vertical, ScannerDown
from neural_robot import unity_frankie as nf
from mm_neo.mm_controllers import MMNeo, VelocityController
from nerf_tools.dataset.nerf_dataset import RosToNeRF



class RosController():
    def __init__(self, initial_pose: sm.SE3, env: swift.Swift, unity , config, scanner , exp_name = "" ,debug = False):
        self.listener = tf.TransformListener()

        self.robot = nf.NeuralFrankie("points_2", spheres = False)
        self.odom_sub = rospy.Subscriber("/frankie/odom", Odometry, self.callback_odom)
        self.unity = unity
        if self.unity: 
            self.joint_state_sub = rospy.Subscriber("/frankie/joint_states", JointState, self.callback_joint_state)
            self.target_poses = rospy.Publisher("/scanner/poses", PoseArray, latch = True, queue_size= 1)
            self.target_poses_msg = PoseArray()
            self.target_poses_msg.poses = []
            self.target_poses_msg.header.frame_id = "map"

            env.set_camera_pose(initial_pose.t + [4.0,0,4.0], initial_pose.t)
            env.step()
            env.add(self.robot)
            self.joint_vel_pub = rospy.Publisher("/frankie/remomac/relay/joint_velocity", Float32MultiArray, queue_size=10)
        else: 
            self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.callback_joint_state) 
            self.joint_vel_pub = rospy.Publisher("/mc/in/joint_velocity", Float32MultiArray, queue_size=10)
            self.target_poses = rospy.Publisher("/scanner/poses", PoseArray, latch = True, queue_size= 1)
            self.target_poses_msg = PoseArray()
            self.target_poses_msg.poses = []
            self.target_poses_msg.header.frame_id = "map"
        
        self.debug = debug
        self.joint_msg = Float32MultiArray()
        
        self.controller_rate = rospy.Rate(200) # 30 Hz
        self.scanning_index = 0
        self.scanning_sequence = scanner
        print(self.scanning_sequence.T_WS_poses)
        self.bridge = CvBridge()
        self.__add_debug_poses(env, self.scanning_sequence.T_WS_poses)
        if self.unity:
            camera_info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo)
        else:
            camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)

        self.joint_vel_controller = VelocityController(self.robot)

        self.dataset = RosToNeRF(f"results/unity_scans/{exp_name}", camera_info, max_depth = 3)
        self.controller = MMNeo(self.robot, config)
        
        self.env = env
        self.qd = np.zeros(9)
        
        
    def step_controller(self):
        arrived = False
        print("step")
        if self.scanning_index >= len(self.scanning_sequence.T_WS_poses):
            self.dataset.save(f"{self.dataset.path}/transforms.json")
            self.qd = np.zeros(9)
            sys.exit(1)
        else:
            target_pose = self.scanning_sequence.get_pose(self.scanning_index)
            arrived, self.qd, _ = self.controller.step(target_pose)
        
        if self.debug:
            self.env.step()
        self.joint_msg.data = self.qd * 0.5
        self.joint_vel_pub.publish(self.joint_msg)
        self.robot.q[:2] = 0
        if arrived:
            rospy.loginfo("Arrived")
            self.qd = np.zeros(9)
            self.joint_msg.data = self.qd * 0.5
            # self.joint_msg.data[0], self.joint_msg.data[1] = self.joint_msg.data[1], self.joint_msg.data[0]
            self.joint_vel_pub.publish(self.joint_msg)
            rospy.sleep(1) 
            self.scanning_index += 1
            if self.unity:
                img = rospy.wait_for_message("/camera/rgb/image", Image)
                depth_image = rospy.wait_for_message("/camera/depth/image", Image)
            else:
                rospy.loginfo("waiting for color image")
                img = rospy.wait_for_message("/camera/color/image_rect_color", Image)
                rospy.loginfo("wait for depth")
                depth_image = rospy.wait_for_message("/camera/depth/image_meters_aligned", Image)
            
            self.save(img, depth_image)
            
        
        return

    def get_camera_pose(self):
        if self.unity:
            T = self.listener.lookupTransform("/map", "/camera_optical_link", rospy.Time(0))
        else:
            T = self.listener.lookupTransform("/map", "/camera_color_optical_frame", rospy.Time(0))
         
        T = sm.SE3.Rt(
            R = smb.q2r(T[1], order="xyzs"),
            t = T[0],
            check = False
        ).norm()
        # T = T.norm() * sm.SE3.Rx(np.pi)
        #
        return T

    def save(self, img, depth_image):
        camera_pose = self.get_camera_pose()
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        self.dataset.record_frame(cv_image, cv_depth_image, camera_pose)


    def callback_odom(self, odom: Odometry):
        # rospy.loginfo("callback_odom")
        base_new = to_spatialmath(odom.pose.pose)
        self.robot._T = base_new.A
        return

    def callback_joint_state(self, joint: JointState):
        # rospy.loginfo("callback_joint_state")
        # TODO Fix this
        if self.unity:
            self.robot.q[2:] = np.array(joint.position[:-1])
            self.robot.grippers[0].q = np.array([joint.position[-1], joint.position[-1]])
        else:
            self.robot.q[2:] = np.array(joint.position[:-2])
            self.robot.grippers[0].q = np.array([joint.position[-2], joint.position[-1]])
 


    def __add_debug_poses(self, env, poses: List[sm.SE3]):
        if self.unity:
            for i in poses:
                axis = sg.Axes(0.1)
                axis.T = i
                env.add(axis)
                print("adding")
            env.step()
            for i in poses:
                pose = to_ros(i)
                self.target_poses_msg.poses.append(pose)
            self.target_poses_msg.header.stamp = rospy.Time.now()    
            self.target_poses.publish((self.target_poses_msg))
            
        else:
            for i in poses:
                pose = to_ros(i)
                self.target_poses_msg.poses.append(pose)
            
            self.target_poses_msg.header.stamp = rospy.Time.now()
            self.target_poses.publish((self.target_poses_msg))
            
            while True:
                text = input("Continue y/n")
                if text == "y":
                    return
                elif text == "n":
                    exit(1)

        return

if __name__ == "__main__":
    
    rospy.init_node("frankie_scanner")

    env = swift.Swift()    
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        exp_name:str = ""
        scanner: VSConfig = VSConfig()
        vertical: bool = False 
        debug: bool = False
        unity: bool = False
    args = tyro.cli(Args)

    rospy.sleep(2)
    
    if args.unity:
        env.launch(realtime= True, headless= not args.debug, browser = "chromium")
        initial_pose = to_spatialmath(rospy.wait_for_message("/frankie/odom", Odometry).pose.pose)
        initial_joint = rospy.wait_for_message("/frankie/joint_states", JointState)
    else:
        initial_pose = to_spatialmath(rospy.wait_for_message("/frankie/odom", Odometry).pose.pose) 
        initial_joint = rospy.wait_for_message("/joint_states", JointState)

    robot = nf.NeuralFrankie("points_8", spheres = False)
    robot.q = robot.qr
    robot.base.T = initial_pose.A 
    initial_eef = robot.fkine(robot.q)

    if args.debug and args.unity:
        from mm_neo.utils.swift_utils import set_camera_robot
        set_camera_robot(env, initial_pose, [4.0,0,4.0])        
        env.step()

    print("Starting scan")

    config = NeoConfig()
    config.home_cost = True

    if args.vertical: 
        scanner = Scanner_vertical(initial_pose, **args.scanner.__dict__)
    else:
        scanner = ScannerDown(initial_pose, 0.3, 0.015, 10, 1, x_offset= initial_eef.t[0] + 0.05, z_offset= initial_eef.t[2])
        config.lamda_q = config.only_base # Penalize the base movement
        # scanner = ScannerDown(robot.base, 0.3, 0.1, N,M, x_offset=eef_initial.t[0], z_offset= eef_initial.t[2])
    Control = RosController(initial_pose, env, args.unity, config, scanner, exp_name = args.exp_name, debug = args.debug)

    while not rospy.is_shutdown():
        Control.step_controller()
        Control.controller_rate.sleep()
    # env.step()
    
