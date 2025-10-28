import numpy as np
import spatialmath as sm
import spatialgeometry as sg
import torch

import rospy

from spatialmath_rospy import to_ros

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Odometry

from spatialmath_rospy import to_spatialmath, to_ros
from visualization_msgs.msg import (
    Marker,
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
)
from interactive_markers.menu_handler import MenuHandler
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from typing import List


def vector_to_pose(position: np.ndarray, vector: np.ndarray) -> Pose:
    """Convert a vector to a pose

    Parameters
    ----------
    position: np.ndarray
        Position of the point [x,y,z]
    vector : np.ndarray
        Gradient estimation of iSDF

    Returns
    -------
    geometry_msgs.Pose
    """
    pose = Pose()

    se_pose = look_at([0, 0, 0], vector)
    pose.position.x, pose.position.y, pose.position.z = position
    theta, v = se_pose.angvec()
    pose.orientation.w = theta
    pose.orientation.x, pose.orientation.y, pose.orientation.z = v
    return pose


def numpy_to_point(position: np.ndarray) -> [Point]:
    """Convert a vector to a point

    Parameters
    ----------
    position: np.ndarray
        Position of the point [x,y,z]

    Returns
    -------
    geometry_msgs.Point
    """
    points = []
    position = position.astype(np.float32)
    for row in position:
        point = Point()
        point.x, point.y, point.z = row
        points.append(point)
    return points


def get_frankie_pose(odom_topic: str = "/frankie/odom") -> sm.SE3:
    """Get the pose of frankie

    Returns
    -------
    sm.SE3
        pose of the base link of frankie in the world frame T_W0
    """
    pose = rospy.wait_for_message(odom_topic, Odometry)
    return to_spatialmath(pose.pose.pose)


def se3_to_ros_pose(pose: sm.SE3) -> Pose:
    """Convert a spatialmath.SE3 to a geometry_msgs.Pose

    Parameters
    ----------
    pose : sm.SE3
        Pose to be converted

    Returns
    -------
    geometry_msgs.Pose
    """
    # ros_pose = Pose()
    # ros_pose.position.x, ros_pose.position.y, ros_pose.position.z = pose.t
    # theta, v = pose.angvec()
    # ros_pose.orientation.w = theta
    # ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z = v

    return to_ros(pose)


def publish_mesh(
    mesh_path: str,
    frame="map",
    pose: sm.SE3 = sm.SE3(),
    topic="sdf_mesh",
    color=[0, 0, 0.55, 1],
):
    marker = mesh_to_marker(mesh_path, frame, pose, color)

    vis_pub = rospy.Publisher(topic, Marker, queue_size=1, latch=True)
    vis_pub.publish(marker)


def publish_point_cloud(publisher: rospy.Publisher, points: np.ndarray, frame="map"):
    """Publish a point cloud to a topic from a tensor

    Parameters
    ----------
    publisher : rospy.Publisher
        publisher object
    points : torch.Tensor
        Tensor with the position of the points to publish, size [N,3]
    frame : str, optional
        Frame where the point cloud is being published, by default "map"
    """
    points = points.astype(np.float32)
    sampling_msg = PointCloud2()
    sampling_msg.header.frame_id = frame
    sampling_msg.header.stamp = rospy.Time.now()
    sampling_msg.height = 1
    sampling_msg.width = points.shape[0]
    sampling_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    sampling_msg.is_bigendian = False
    sampling_msg.point_step = 12
    sampling_msg.row_step = sampling_msg.point_step * sampling_msg.width
    sampling_msg.is_dense = True
    sampling_msg.data = points.tobytes()
    publisher.publish(sampling_msg)


def tensor_to_pointcloud(points: torch.Tensor, frame="map") -> PointCloud2:
    """Convert a tensor to a point cloud message

    Parameters
    ----------
    points : torch.Tensor
        Tensor with the position of the points to publish, size [N,3]
    frame : str, optional
        Frame where the point cloud is being published, by default "map"

    Returns
    -------
    PointCloud2
    """
    # print("processing sampled points")
    points = points.type(torch.float32)
    sampling_msg = PointCloud2()
    sampling_msg.header.frame_id = frame
    sampling_msg.header.stamp = rospy.Time.now()
    sampling_msg.height = 1
    sampling_msg.width = points.shape[0]
    sampling_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    sampling_msg.is_bigendian = False
    sampling_msg.point_step = 12
    sampling_msg.row_step = sampling_msg.point_step * sampling_msg.width
    sampling_msg.is_dense = True
    sampling_msg.data = points.clone().detach().cpu().numpy().tobytes()
    return sampling_msg


def mesh_to_marker(
    mesh_path, frame="map", pose: sm.SE3 = sm.SE3(), color=[0, 0, 0.55, 1]
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame
    marker.type = marker.MESH_RESOURCE
    marker.mesh_resource = f"file://{mesh_path}"
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.scale.x, marker.scale.y, marker.scale.z = 1.0, 1.0, 1.0
    marker.pose = to_ros(pose)
    print("mesh pose", pose)

    marker.header.stamp = rospy.Time.now()
    return marker


def makeBox(msg):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.CUBE
    marker.color.a = 1.0
    marker.color.b = 0.55
    marker.scale.x, marker.scale.y, marker.scale.z = 1.0, 1.0, 1.0
    marker.pose = to_ros(sm.SE3())

    marker.header.stamp = rospy.Time.now()
    return marker


def makeBoxControl(msg):
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(msg))
    msg.controls.append(control)
    return control


class Interactive6DOFPose:
    def __init__(self, initial_pose: sm.SE3(), topic: str, meshes: List[sg.Mesh]):
        self.server = InteractiveMarkerServer(topic)
        self.menu_handler = MenuHandler()
        self.pose = initial_pose
        marker = InteractiveMarker()
        marker.header.frame_id = "map"
        marker.pose = to_ros(initial_pose)

        marker.scale = 1.0
        marker.name = topic
        marker.description = "6DOF Pose"

        self.marker = marker

        for mesh in meshes:
            control = InteractiveMarkerControl()
            control.always_visible = True
            control.markers.append(
                mesh_to_marker(
                    mesh.filename, "", sm.SE3(mesh.T), color=[0, 0.55, 0, 0.7]
                )
            )
            self.marker.controls.append(control)
        # self.marker.markers.append(load_mesh(mesh.filename, "", sm.SE3(mesh.T)))
        # self.marker.controls[0].interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        self.marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        self.marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        self.marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        self.marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        self.marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        self.marker.controls.append(control)

        self.server.insert(self.marker, self.processFeedback)
        self.menu_handler.apply(self.server, self.marker.name)
        self.server.applyChanges()

    def set_pose(self, pose: sm.SE3):
        self.pose = pose
        self.marker.pose = to_ros(pose)
        self.server.insert(self.marker, self.processFeedback)
        self.server.applyChanges()

    def get_pose(self) -> sm.SE3:
        return self.pose

    def processFeedback(self, feedback):
        self.pose = to_spatialmath(feedback.pose)
        s = "Feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"

        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id

        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo(s + ": button click" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            rospy.loginfo(
                s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "."
            )
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo(s + ": pose changed")
        self.server.applyChanges()


if __name__ == "__main__":
    rospy.init_node("test_interactive_marker")
    from mm_neo.utils.swift_utils import load_mesh, load_gripper_frankie
    from neural_robot.neural_frankie import NeuralFrankie

    robot = NeuralFrankie("points_2")
    robot.base = get_frankie_pose()
    robot.q = robot.qr

    gripper = load_gripper_frankie(robot)

    Interactive6DOFPose(robot.fkine(robot.q), "test", gripper)
    rospy.spin()
