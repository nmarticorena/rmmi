import numpy as np
import spatialmath as sm
from spatialmath_rospy import to_spatialmath, to_ros

from typing import Optional

# Ros messages
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray
from mm_neo.configs.controllers import NeoConfig

from mm_neo.mm_controllers import VelocityController, MMNeoSDF, GripperController
import neural_robot.unity_frankie as nf

from nerf_tools.dataset.nerf_dataset import RosToNeRF

from mm_neo.utils.ros_utils import publish_mesh, Interactive6DOFPose
from mm_neo.utils.swift_utils import load_gripper_frankie
from mm_neo.sdf import SDF
from mm_neo.logger import Logger
from mm_neo.omron import Omron

from armer_msgs.msg import ManipulatorState

try:
    from rv_msgs.msg import JointVelocity
except ImportError:
    print("Cannot import rv working on local computer")


class RobotDebugger:
    def __init__(
        self,
        odom_topic,
        joint_topic,
        color=[0, 1, 0],
        robot_name: str = "post_new_points_3",
        spheres: bool = False,
    ):
        self.qd = np.zeros(9)
        self.robot = nf.NeuralFrankie(robot_name, spheres=spheres)
        self.robot.replace_point_meshes(color, True)
        self.odom_topic = odom_topic
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.callback_odom, queue_size=1
        )
        self.joint_state_sub = rospy.Subscriber(
            joint_topic, JointState, self.callback_joint_state, queue_size=1
        )
        self.target_pose_sub = rospy.Subscriber(
            "/frankie/target_pose",
            PoseStamped,
            self.callback_target,
        )
        self.target = sm.SE3()

    def callback_target(self, pose: PoseStamped):
        pose = to_spatialmath(pose.pose)
        self.target = pose

    def callback_odom(self, odom: Odometry):
        # rospy.loginfo(f"callback_odom {self.odom_topic}")
        base_new = to_spatialmath(odom.pose.pose)
        self.robot._T = base_new.A
        self.robot.base = base_new.A
        self.qd[0] = odom.twist.twist.linear.x
        self.qd[1] = odom.twist.twist.angular.z
        return

    def callback_joint_state(self, joint: JointState):
        # rospy.loginfo("callback_joint_state")
        self.robot.q[2:] = np.array(joint.position[:-2])
        self.qd[2:] = np.array(joint.velocity[:-2])
        return


class UnitySubscriber:
    def __init__(self, robot_name: str = "curobo"):
        if "curobo" in robot_name:
            self.robot = nf.NeuralFrankie("curobo", spheres=True)
        else:
            self.robot = nf.NeuralFrankie(robot_name)
        self.odom_sub = rospy.Subscriber("/frankie/odom", Odometry, self.callback_odom)
        self.joint_state_sub = rospy.Subscriber(
            "/frankie/joint_states", JointState, self.callback_joint_state
        )

    def callback_odom(self, odom: Odometry):
        # rospy.loginfo("callback_odom")
        base_new = to_spatialmath(odom.pose.pose)
        self.robot._T = base_new.A
        return

    def callback_joint_state(self, joint: JointState):
        # rospy.loginfo("callback_joint_state")
        self.robot.q[2:] = np.array(joint.position[:-1])
        self.robot.qd[2:] = np.array(joint.velocity[:-1])
        return


class RosInterface:
    def __init__(
        self,
        config: Optional[NeoConfig] = NeoConfig(),
        robot_name: str = "curobo_2",
        spheres: bool = True,
    ):
        self.robot = nf.NeuralFrankie(robot_name, spheres=spheres)
        print("Robot name", robot_name)
        self.robot.qdlim[0] = 0.5  # Vx 1 m/s
        self.robot.qdlim[1] = 0.87  # Wz 50 Deg/s
        self.last_qd = np.zeros(9)
        print(len(self.robot.Tpr_i))
        self.qdd_lim_base = np.array([0.15, 0.15])
        self.qdd_lim_arm = np.ones(7) * 1

        self.qdd_lim = np.concatenate([self.qdd_lim_base, self.qdd_lim_arm])
        # self.odom_sub = rospy.Subscriber("/frankie/odom/interpolated", Odometry, self.callback_odom)
        self.odom_sub = rospy.Subscriber("/frankie/odom", Odometry, self.callback_odom)
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states", JointState, self.callback_joint_state
        )
        self.joint_vel_pub = rospy.Publisher(
            "/mc/in/joint_velocity", Float32MultiArray, queue_size=10, tcp_nodelay=True
        )
        self.joint_msg = Float32MultiArray()
        self.arm_status_sub = rospy.Subscriber(
            "/arm/state", ManipulatorState, self.callback_arm_status
        )
        self.arm_status = ManipulatorState()

        # Debug publishers
        self.ee_pose_pub = rospy.Publisher(
            "/frankie/ee_pose", PoseStamped, queue_size=10
        )
        self.ee_pose = PoseStamped()
        self.ee_pose.header.frame_id = "map"
        self.target_ee_pose_pub = rospy.Publisher(
            "/frankie/target_pose", PoseStamped, queue_size=10
        )
        self.target_ee_pose = PoseStamped()
        self.target_ee_pose.header.frame_id = "map"

        self.controller_rate = rospy.Rate(config.control_frecuency)
        self.scanning_index = 0
        self.bridge = CvBridge()

        self.joint_vel_controller = VelocityController(self.robot)
        self.gripper_controller = GripperController(self.robot.grippers[0])
        self.controller = MMNeoSDF(self.robot, config)
        self.vel_controller = VelocityController(self.robot)
        self.qd = np.zeros(9)

    def stop(self):
        self.joint_msg.data = np.zeros(9)
        self.joint_vel_pub.publish(self.joint_msg)
        self.qd = np.zeros(9)

    def step(self):
        self.joint_msg.data = self.qd

        self.joint_msg.data[0], self.joint_msg.data[1] = (
            self.joint_msg.data[1],
            self.joint_msg.data[0],
        )
        self.joint_vel_pub.publish(self.joint_msg)
        self.controller_rate.sleep()
        self.ee_pose.pose = to_ros(self.robot.fkine(self.robot.q))
        self.ee_pose_pub.publish(self.ee_pose)

    def get_frame(self) -> tuple[Image, Image, sm.SE3]:
        img = rospy.wait_for_message("/cam", Image)
        depth_image = rospy.wait_for_message("/depth_cam", Image)
        camera_pose = to_spatialmath(
            rospy.wait_for_message("/camera_pose", PoseStamped).pose
        ) * sm.SE3.Rx(np.pi)
        return img, depth_image, camera_pose

    def callback_arm_status(self, arm_status: ManipulatorState):
        self.arm_status = arm_status
        return

    def callback_odom(self, odom: Odometry):
        vx = odom.twist.twist.linear.x
        wz = odom.twist.twist.angular.z
        self.qd_base = np.array([vx, wz])
        # rospy.loginfo("callback_odom")
        base_new = to_spatialmath(odom.pose.pose)
        self.robot._T = base_new.A
        return

    def callback_joint_state(self, joint: JointState):
        # TODO add a flag when is unity or real robot
        # rospy.loginfo("callback_joint_state")
        # TODO Migth be interesting to do this

        self.robot.q[2:] = np.array(joint.position[:-2])
        alpha = 1.0
        self.robot.q[2:] = (
            alpha * np.array(joint.position[:-2]) + (1 - alpha) * self.robot.q[2:]
        )

        self.last_qd_arm = np.array(joint.velocity[:-2])
        # self.robot.qd[2:] = np.array(joint.velocity[:-1])
        # print("desire gripper", joint.position[-1])
        self.robot.grippers[0].q = np.array([0, 0])


class Scanner(RosInterface):
    def __init__(self, initial_pose: sm.SE3, config):
        super().__init__(initial_pose, config)
        camera_info = rospy.wait_for_message("/camera_info", CameraInfo)
        self.dataset = RosToNeRF("results/unity_scans", camera_info)

    def save_frame(self):
        img, depth_image, camera_pose = self.get_frame()
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        cv_depth_image = self.bridge.imgmsg_to_cv2(
            depth_image, desired_encoding="passthrough"
        )
        self.dataset.record_frame(cv_image, cv_depth_image, camera_pose)

    def compute_vel(self):
        arrived = False
        done = False

        print("step")
        if self.scanning_index >= len(self.scanning_sequence.T_WS_poses):
            self.dataset.save("transforms.json")
            qd = np.zeros(9)
            done = True
        else:
            target_pose = self.scanning_sequence.get_pose(self.scanning_index)
            arrived, qd = self.controller.step(target_pose)

        self.robot.q[:2] = 0
        if arrived:
            self.scanning_index += 1
            self.save_frame()
        return done, qd


class CollisionAvoidance(RosInterface):
    def __init__(
        self,
        config: NeoConfig,
        target_poses: list[sm.SE3],
        sdf: SDF,
        mesh_file: str,
        robot_name: str = "post_new_points_3",
        spheres: bool = False,
        logger: Optional[Logger] = None,
        max_attempts: int = 1500,
    ):
        super().__init__(config, robot_name, spheres)
        self.base = Omron()

        self.n_smooth = 1
        rospy.wait_for_message("/joint_states", JointState)
        # rospy.wait_for_message("/frankie/odom", Odometry)
        rospy.wait_for_message("/frankie/odom/interpolated", Odometry)
        self.q_smooth = np.zeros((self.n_smooth, 9))
        self.q_smooth_index = 0
        self.status_pub = rospy.Publisher("/status", Float32MultiArray, queue_size=10)
        self.status_msg = Float32MultiArray()
        self.status_msg.data = np.zeros(2)
        self.qd_debug_pub = rospy.Publisher(
            "/joint_velocity_debug", Float32MultiArray, queue_size=30
        )
        self.qd_debug = Float32MultiArray()

        self.max_attempts = max_attempts  # approx 10 seconds
        self.attempts = 0
        self.smooth_pose = 0  # [0,1]
        self.smooth_step = 0.15
        self.target_poses = target_poses
        self.sdf = sdf
        self.collision_controller = MMNeoSDF(self.robot, sdf, config, logger=logger)
        publish_mesh(mesh_file)
        self.target_index = 0
        gripper = load_gripper_frankie(self.robot)
        self.target_pose_viz_pub = rospy.Publisher(
            "/target_poses", PoseArray, queue_size=10, latch=True
        )

        self.interactive_pose = Interactive6DOFPose(
            sm.SE3(target_poses[0]), "eof", gripper
        )
        self.target_poses = target_poses
        self.target_pose = self.target_poses[0]
        self.previous_pose = self.target_poses[0]
        self.visualize_target_poses()

    def visualize_target_poses(self):
        targets = PoseArray()
        targets.header.frame_id = "map"
        for pose in self.target_poses:
            target = to_ros(pose)
            targets.poses.append(target)
        self.target_pose_viz_pub.publish(targets)

    def set_status(self, status: int, exp_n: int):
        """
        Set the status of the experiment
        Parameters
        ----------
        status: int
            Status of the experiment [0: waiting, 1: running, 2: base, 3: arm, 4: prepose]
        exp_n: int
            Experiment number
        """
        self.status_msg.data = [status, exp_n]
        self.status_pub.publish(self.status_msg)

    def smooth_poses(self, current_pose: sm.SE3, new_target: sm.SE3):
        if self.smooth_pose >= 1:
            return new_target
        return current_pose.interp(new_target, self.smooth_pose)

    def select_target(self, idx):
        """
        Select the next target pose
        Parameters
        ----------
        idx: int
            Index of the target pose
        """
        rospy.loginfo(f"Selecting target {idx}")
        self.previous_pose = self.target_pose.copy()
        if idx >= len(self.target_poses):
            idx = len(self.target_poses) - 1
            print("Index out of range, goint to last one")
        self.target_pose = self.target_poses[idx].copy()
        self.target_ee_pose.pose = to_ros(self.target_pose)
        self.target_ee_pose_pub.publish(self.target_ee_pose)

        self.qd = np.zeros(9)
        self.q_smooth = np.zeros((self.n_smooth, 9))
        self.smooth_pose = 0
        self.q_smooth_index = 0

    def pre_pose(self, offset: sm.SE3 = sm.SE3(0, -0.5, 0)):
        """
        Add a new target
        Parameters
        ----------
        offset: sm.SE3
            New target pose
        """
        self.previous_pose = self.target_pose.copy()
        self.target_pose = offset * self.target_pose
        self.target_ee_pose.pose = to_ros(self.target_pose)
        self.target_ee_pose_pub.publish(self.target_ee_pose)

        self.qd = np.zeros(9)
        self.q_smooth = np.zeros((self.n_smooth, 9))
        self.smooth_pose = 0
        self.q_smooth_index = 0

    def stop(self):
        rospy.loginfo("Stopping the robot")
        self.qd = np.zeros(9)
        self.joint_msg.data = np.zeros(9)
        self.joint_vel_pub.publish(self.joint_msg)
        self.q_smooth = np.zeros((self.n_smooth, 9))
        self.smooth_pose = 0
        self.target_index = 0
        self.attempts = 0
        return

    def step_pose(self):
        """
        Comute the target velocity to reach the target pose
        # TODO add the log of the error or final

        We can end due to timeout ,reach the target or solver
        failed

        Returns
        -------
        bool
            True if the target is reached
        dict
            debug info
        """
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            self.stop()
            rospy.loginfo("Timeout")
            return True, "Timeout"

        current_target = self.smooth_poses(self.previous_pose, self.target_pose)
        if current_target is None:
            current_target = self.target_pose
        arrived, qd, failed = self.collision_controller.step(current_target)
        self.qd_debug.data = self.collision_controller.debug_qd
        self.qd_debug_pub.publish(self.qd_debug)
        if not self.collision_controller.config.collisions:
            error = self.arm_status.errors
            if error == 1 or error == 2:
                rospy.loginfo(f"Error {error}")
                rospy.loginfo("Collision detected")
                return True, "failed"
        if failed:
            print("Failed")
            self.stop()
            return True, "failed"

        self.q_smooth_index += 1
        self.q_smooth_index = self.q_smooth_index % self.n_smooth
        self.q_smooth[self.q_smooth_index] = qd
        self.smooth_pose += self.smooth_step
        if arrived and self.smooth_pose >= 1:
            rospy.loginfo("Arrived")
            self.stop()
            return True, "Arrived"
        else:
            self.qd = self.q_smooth.mean(axis=0)
            return False, "Running"
