import copy
import sys
import numpy as np
from spatialmath_rospy import to_spatialmath
import swift
import spatialgeometry as sg
import spatialmath as sm

# Ros stuff
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState, CameraInfo
from mm_neo.configs.controllers import NeoConfig
from mm_neo.configs.sdf import VSConfig

# custom libraries
from mm_neo.scanner import Scanner_vertical, ScannerDown
from neural_robot import neural_frankie as nf
from nerf_tools.dataset.nerf_dataset import RosToNeRF
import tf
import spatialmath.base as smb


class DummyScanner():
    def __init__(self,n_frames , exp_name = ""):
        
        
        self.bridge = CvBridge()
        #TODO Should be color or depth?
        camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.listener = tf.TransformListener()
        self.dataset = RosToNeRF(f"results/real_scans/{exp_name}", camera_info)
        self.n_frames = n_frames 
        self.index = 0

    def get_camera_pose(self):
        T = self.listener.lookupTransform("/map", "/camera_color_optical_frame", rospy.Time(0))
         
        T = sm.SE3.Rt(
            R = smb.q2r(T[1], order="xyzs"),
            t = T[0],
            check = False
        )
        T = T.norm()

        return T

    def step(self):
        input(f"Next Frame {self.index}/{self.n_frames}")     
        rospy.loginfo("waiting image")
        img = rospy.wait_for_message("/camera/color/image_rect_color", Image)
        rospy.loginfo("waiting depth")
        depth_image = rospy.wait_for_message("/camera/depth/image_meters_aligned", Image)
        self.save(img, depth_image)
        self.index += 1 

        if self.index >= self.n_frames:
            self.dataset.save(f"{self.dataset.path}/transforms.json")
            sys.exit(1)
            
        
        return

    def save(self, img, depth_image):
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        rospy.loginfo("Waiting for camera pose")
        camera_pose = self.get_camera_pose()  * sm.SE3.Rx(np.pi)

        self.dataset.record_frame(cv_image, cv_depth_image, camera_pose)


if __name__ == "__main__":
    
    rospy.init_node("frankie_scanner")

    import tyro
    from dataclasses import dataclass
    @dataclass
    class Args:
        n_frames:int = 10
        exp_name:str = "ros_test"
        
    args = tyro.cli(Args)
    rospy.sleep(2)
    
    Scanner = DummyScanner(args.n_frames, args.exp_name)

    while not rospy.is_shutdown():
        Scanner.step() 
