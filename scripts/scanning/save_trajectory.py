import numpy as np
import os
from spatialmath_rospy import to_spatialmath
import spatialmath as sm
import spatialmath.base as smb

# Ros stuff
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class TrajRecorder():
    def __init__(self, pose_topic ,n_frames:int = 60 ,
                 exp_name = "", period_time: float = 1, 
                 append: bool = False):
        self.pose_topic = pose_topic 
        self.frame_index = 0
        self.n_frames = n_frames

        self.last_time = rospy.Time.now()
        
        self.camera_pose_sub = rospy.Subscriber(self.pose_topic, Odometry, self.callback)

        self.period_time = rospy.Duration(period_time)
        self.last_frame_time = rospy.Time.now()
        os.makedirs("results/unity_traj/", exist_ok= True) 
        self.poses = []
        self.exp_name = exp_name
        self.append = append

    def callback_cmd_vel(self, cmd_vel: Twist):
        self.cmd_vel = cmd_vel
        self.last_time = rospy.Time.now()
        return

    def callback(self, robot_pose:Odometry):
        self.robot_pose:sm.SE3 = to_spatialmath(robot_pose.pose.pose)
        # rospy.loginfo(self.robot_pose.t)
        # rospy.loginfo(self.robot_pose.R.)
        return
        
    def save(self):
        if rospy.Time.now() - self.last_frame_time < self.period_time:
            return False
       
        rospy.loginfo("saving position")
        robot_pose = to_spatialmath(rospy.wait_for_message(self.pose_topic, Odometry).pose.pose)
        
        t = robot_pose.t
        rpy = smb.r2q(robot_pose.R, order = "xyzs")
        self.poses.append([*t, *rpy]) 
        


        # pdb.set_trace()
        self.frame_index += 1
        finished = self.frame_index > self.n_frames
        
        open_mode = "a" if self.append else "w"

        if finished:
            with open("results/unity_traj/"+self.exp_name + ".txt", open_mode) as f:
                np.savetxt(f, self.poses) 
        
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
        n_frames:int = 30
        time: float = 1
        append: bool = False 
    args = tyro.cli(Args)



    # rate = rospy.Rate(1 / args.time)
    recorder = TrajRecorder("/frankie/odom", 
                        exp_name = args.exp_name, 
                        n_frames = args.n_frames, 
                        period_time = args.time,
                        append = args.append)   
    
    ti = time.time()
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        if recorder.save():
            break
        
        # recorder.save()
    # env.step()
    
