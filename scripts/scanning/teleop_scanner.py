import numpy as np
from spatialmath_rospy import to_spatialmath
import spatialmath as sm

# Ros stuff
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer

# custom libraries
from nerf_tools.dataset.nerf_dataset import RosToNeRF

import multiprocessing  as mp

class Recorder():
    def __init__(self, pose_topic ,n_frames:int = 60 ,exp_name = "", period_time: float = 1):
        self.pose_topic = pose_topic 
        self.bridge = CvBridge()
        camera_info = rospy.wait_for_message("/camera_info", CameraInfo)
        self.dataset = RosToNeRF(f"results/unity_scans/{exp_name}", camera_info)
        self.frame_index = 0
        self.n_frames = n_frames

        self.last_time = rospy.Time.now()
        
        image_sub       = Subscriber("/cam", Image)
        depth_image_sub = Subscriber("/depth_cam", Image)
        camera_pose_sub = Subscriber(self.pose_topic, PoseStamped)

        cmd_vel_sub = rospy.Subscriber("/keyboard/teleop", Twist, self.callback_cmd_vel)
        self.cmd_vel_pub = rospy.Publisher("/frankie/cmd_vel", Twist, queue_size=1)
        
        self.cmd_vel = Twist()
        
        ts = ApproximateTimeSynchronizer([image_sub, depth_image_sub, camera_pose_sub], 100, 0.1)
        ts.registerCallback(self.callback)
        
        self.period_time = rospy.Duration(period_time)
        self.last_frame_time = rospy.Time.now()

    def callback_cmd_vel(self, cmd_vel: Twist):
        self.cmd_vel = cmd_vel
        self.last_time = rospy.Time.now()
        return

    def callback(self, img, depth_image, camera_pose):
        print("callback")
        self.img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        self.camera_pose = to_spatialmath(camera_pose.pose) * sm.SE3.Rx(np.pi)
        return

    def step(self):
        if rospy.Time.now() - self.last_time > rospy.Duration(1):
            cmd_vel = Twist()
        elif rospy.Time.now() - self.last_frame_time > self.period_time:
            cmd_vel = Twist()
        else:
            cmd_vel = self.cmd_vel
        self.cmd_vel_pub.publish(cmd_vel)
        
        

    def save(self):
        if rospy.Time.now() - self.last_frame_time < self.period_time:
            return False
        rospy.sleep(0.01)  
        img = rospy.wait_for_message("/cam", Image)
        depth_image = rospy.wait_for_message("/depth_cam", Image)
        camera_pose = to_spatialmath(rospy.wait_for_message(self.pose_topic, PoseStamped).pose) * sm.SE3.Rx(np.pi)
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        
        
        self.dataset.record_frame(img, depth_image, camera_pose)
        self.frame_index += 1
        finished = self.frame_index > self.n_frames
        if finished:
            self.dataset.save(f"{self.dataset.path}/transforms.json")
        
        self.last_frame_time = rospy.Time.now()    
        return self.frame_index > self.n_frames
    
    
if __name__ == "__main__":
    
    rospy.init_node("frankie_scanner")

    import tyro
    from dataclasses import dataclass
    import time
    @dataclass
    class Args:
        exp_name:str = ""
        n_frames:int = 60
        time: float = 1
        
    args = tyro.cli(Args)



    # rate = rospy.Rate(1 / args.time)
    recorder = Recorder("/camera_pose", 
                        exp_name = args.exp_name, 
                        n_frames = args.n_frames, 
                        period_time = args.time)   
    
    ti = time.time()
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        recorder.step()
        if recorder.save():
            break
        
        # recorder.save()
    # env.step()
    
