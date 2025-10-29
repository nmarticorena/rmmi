import numpy as np
from spatialmath_rospy import to_spatialmath
import spatialmath as sm

# Ros stuff
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, CameraInfo

# custom libraries
from nerf_tools.dataset.nerf_dataset import RosToNeRF

#Unity service
from unity_robotics_demo_msgs.srv import PositionService, PositionServiceRequest, PositionServiceResponse


class Recorder():
    def __init__(self, pose_topic , skip_frames = 1, traj_name = "", exp_name = ""):
        self.pose_topic = pose_topic
        rospy.loginfo("Waiting for /camera_info")
        camera_info = rospy.wait_for_message("/camera/camera_info", CameraInfo)
        self.dataset = RosToNeRF(f"results/unity_scans/{exp_name}", camera_info)
        self.frame_index = 0
        self.bridge = CvBridge()
        self.robot_poses = np.loadtxt(f"results/unity_traj/{traj_name}.txt")
        self.skip_frames = skip_frames 
        rospy.loginfo("waiting for service")
        rospy.wait_for_service("/frankie_pose_srv")
        self.pose_service = rospy.ServiceProxy('/frankie_pose_srv', PositionService)

        return

    def callback(self, img, depth_image, camera_pose):
        print("callback")
        self.img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        self.camera_pose = to_spatialmath(camera_pose.pose) * sm.SE3.Rx(np.pi)
        return 

        

    def save(self):
        if self.frame_index >= self.robot_poses.shape[0]:
            self.dataset.save(f"{self.dataset.path}/transforms.json")
            return True
       

        x,y,z,rx,ry,rz,w = self.robot_poses[self.frame_index,:]
        new_pose = PositionServiceRequest()
        new_pose.input.pos_x = x
        new_pose.input.pos_y = z # Rigth hand to left hand
        new_pose.input.pos_z = y
        new_pose.input.rot_x = rx
        new_pose.input.rot_y = -rz # Rigth hand to left hand
        new_pose.input.rot_z = ry
        new_pose.input.rot_w = w


        
        self.pose_service(new_pose)

        rospy.sleep(1)

        rospy.loginfo("Waiting for /cam")
        img = rospy.wait_for_message("/camera/image", Image)
        rospy.loginfo("Waiting for /depth_cam")
        depth_image = rospy.wait_for_message("/camera/depth_image", Image)
        rospy.loginfo(f"waiting for {self.pose_topic}")
        camera_pose = to_spatialmath(rospy.wait_for_message(self.pose_topic, PoseStamped).pose) * sm.SE3.Rx(np.pi)
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        
        
        self.dataset.record_frame(img, depth_image, camera_pose)
        self.frame_index += self.skip_frames
        
        return False 
    
if __name__ == "__main__":
    
    rospy.init_node("frankie_scanner")

    import tyro
    from dataclasses import dataclass
    import time
    @dataclass
    class Args:
        exp_name:str = ""
        traj_name:str = ""
        skip_frames:int = 1
        
    args = tyro.cli(Args)



    # rate = rospy.Rate(1 / args.time)
    recorder = Recorder("/camera_pose",
                        skip_frames = args.skip_frames,
                        traj_name= args.traj_name,
                        exp_name = args.exp_name)
                        
    ti = time.time()
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        if recorder.save():
            break
        
        # recorder.save()
    # env.step()
    
