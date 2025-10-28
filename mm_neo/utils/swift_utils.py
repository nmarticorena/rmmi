"""
This module contains functions usefull on swift
"""

from neural_robot.neural_frankie import NeuralFrankie
import swift
import spatialgeometry as sg
import spatialmath as sm
import numpy as np
from typing import List


def set_camera_robot(env: swift.Swift, T_Wb, p_bc):
    """Set the camera pose to look at the robot base with a displacement of camera_pose

    Parameters
    ----------
    env : swift.Swift
        kinematic simulator
    T_Wb : sm.SE3
        Pose of the robot base link with respect to the world T_W0
    p_bc : np.ndarray
        Camera position
    """
    T_Wc = T_Wb * sm.SE3(p_bc)
    env.set_camera_pose(T_Wc.t, T_Wb.t)

def look_at(origin: np.ndarray, towards: np.ndarray, with_translation=False) -> sm.SE3:
    """
    Generate a SE3 pose that looks at a point

    Parameters
    ----------
    origin : np.ndarray
        [x,y,z]
    towards : array
        target [x,y,z]
    with_translation: bool
        Flag to add translation of the origin to the pose
    """
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function

    forward = towards - origin  # Normalized
    forward = forward / np.linalg.norm(forward)
    # Step 2  compute rigth axis
    right = np.cross([1.0, 0.0, 0.0], forward)  # 1,0,0
    right = right / np.linalg.norm(right)
    # Step 3 compute up axis
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Step 4 compute rotation matrix

    pose_matrix = np.eye(4)

    pose_matrix[0][0] = right[0]
    pose_matrix[1][0] = right[1]
    pose_matrix[2][0] = right[2]
    pose_matrix[0][1] = up[0]
    pose_matrix[1][1] = up[1]
    pose_matrix[2][1] = up[2]
    pose_matrix[0][2] = forward[0]
    pose_matrix[1][2] = forward[1]
    pose_matrix[2][2] = forward[2]

    if with_translation:
        pose_matrix[0][3] = origin[0]
        pose_matrix[1][3] = origin[1]
        pose_matrix[2][3] = origin[2]

    pose_matrix[3][3] = 1.0

    pose = sm.SE3(pose_matrix, check=False).norm()

    return pose

def load_mesh(filename: str, scale=1.0, color=[1, 0.4, 0, 0.5]) -> sg.Mesh:
    """Load a mesh from a file

    Args:
        filename (str): Path to mesh file
        scale (float, optional): Scale of the mesh. Defaults to 1.0.
        color (list, optional): Color of the mesh. Defaults to [1,0.4,0,0.5].
    Returns:
        sg.Mesh: spatial object of the loaded mesh ready to be loaded to swift
    """
    final_scale = [scale, scale, scale]
    mesh = sg.Mesh(filename, scale=final_scale, color=color)
    return mesh


class interactive_point:
    def __init__(
        self,
        env: swift.Swift,
        initial_pose: sm.SE3 = sm.SE3(),
        radius=0.01,
        colour=[1, 0, 0, 1],
        range=1,
    ) -> None:
        self.sphere = sg.Axes(radius, pose=initial_pose)
        self.X, self.Y, self.Z = initial_pose.t

        env.add(self.sphere)

        x_initial = initial_pose.t[0]
        y_initial = initial_pose.t[1]
        z_initial = initial_pose.t[2]

        self.x_slider = swift.Slider(
            self.x_callback, -range + x_initial, range + x_initial, 0.01, self.X, "x"
        )
        self.y_slider = swift.Slider(
            self.y_callback, -range + y_initial, range + y_initial, 0.01, self.Y, "y"
        )
        self.z_slider = swift.Slider(
            self.z_callback, -range + z_initial, range + z_initial, 0.01, self.Z, "z"
        )

        env.add(self.x_slider)
        env.add(self.y_slider)
        env.add(self.z_slider)

    def update_pose(self):
        self.sphere.T = sm.SE3(self.X, self.Y, self.Z)
        return

    def x_callback(self, value):
        self.X = value
        self.update_pose()

    def y_callback(self, value):
        self.Y = value
        self.update_pose()

    def z_callback(self, value):
        self.Z = value
        self.update_pose()

class interative_pose(interactive_point):
    def __init__(
        self, env, initial_pose: sm.SE3, radius=0.1, colour=[1, 0, 0, 1], range=1
    ) -> None:
        super().__init__(env, initial_pose, radius, colour, range)

        self.X, self.Y, self.Z = initial_pose.t
        self.phi, self.theta, self.psi = initial_pose.rpy()

        self.phi_slider = swift.Slider(
            self.phi_callback, -np.pi, np.pi, np.pi / 20, self.phi, "Rz"
        )
        self.theta_slider = swift.Slider(
            self.theta_callback, -np.pi, np.pi, np.pi / 20, self.theta, "Ry"
        )
        self.psi_slider = swift.Slider(
            self.psi_callback, -np.pi, np.pi, np.pi / 20, self.psi, "Rx"
        )
        env.add(self.phi_slider)
        env.add(self.theta_slider)
        env.add(self.psi_slider)

    def phi_callback(self, value):
        self.phi = value
        self.update_pose()

    def theta_callback(self, value):
        self.theta = value
        self.update_pose()

    def psi_callback(self, value):
        self.psi = value
        self.update_pose()

    def update_pose(self):
        self.sphere.T = sm.SE3(self.X, self.Y, self.Z) * sm.base.rpy2tr(
            self.phi, self.theta, self.psi
        )

class interative_mesh(interative_pose):
    def __init__(
        self, env, meshes: List[sg.Mesh], initial_pose, pose_to_mesh, range=1.0
    ):
        super().__init__(env, initial_pose, range=range)
        self.meshes = meshes
        self.pose_to_mesh = pose_to_mesh

        for ix, mesh in enumerate(meshes):
            # mesh.attach_to(self.sphere)
            # mesh.T = self.pose_to_mesh[ix]
            env.add(mesh)

    def update_pose(self):
        super().update_pose()
        for ix, mesh in enumerate(self.meshes):
            mesh.T = self.sphere.T * self.pose_to_mesh[ix]

class interative_panda_gripper(interative_mesh):
    def __init__(self, env, robot: NeuralFrankie, range=3):
        link_6 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link6.stl")
        link_7 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link7.stl")
        hand = load_mesh(f"{robot.config_path}/meshes/frankie/frankie_hand.stl")
        meshes = [link_6, link_7, hand]

        hand_pose = sm.SE3(robot.links[-1].ets[0].A())
        link_7_pose = robot.fkine(robot.q, start=robot.links[-1], include_base=False)
        # for ets in robot.links[-2].ets:
        #     link_7_pose = sm.SE3(ets.A()) * link_7_pose
        link_7_pose = link_7_pose  # * hand_pose
        # link_7_pose = sm.SE3(robot.links[-2].ets[0].A()) * hand_pose
        link_6_pose = robot.fkine(robot.q, start=robot.links[-2], include_base=False)

        pose_to_mesh = [link_6_pose.inv(), link_7_pose.inv(), hand_pose.inv()]

        super().__init__(env, meshes, robot.fkine(robot.q), pose_to_mesh, range=range)

class moving_point(interactive_point):
    def __init__(
        self,
        env: swift.Swift,
        start_pose=sm.SE3(),
        radius=0.01,
        colour=[1, 0, 0, 1],
        range=5,
    ) -> None:
        super().__init__(env, start_pose, radius, colour, range)

        self.state = False

        # Add button to start the movement
        self.button = swift.Button(self.button_callback, "Move")
        env.add(self.button)
        # Add slider to control the speed
        self.speed = 0.001
        self.speed_slider = swift.Slider(self.speed_callback, 0, 1, 0.001, 1, "speed")
        env.add(self.speed_slider)

    def speed_callback(self, value):
        self.speed = value

    def button_callback(self, x):
        print("button callbakc")
        self.state = not self.state
        print("state", self.state)

    def step(self, gradient, distance):
        vel = -gradient[0] * self.speed
        if self.state and distance > 0.001:
            print("Moving")
            self.sphere.T = self.sphere.T * sm.SE3(vel[0], vel[1], vel[2])

def load_gripper_frankie(robot):
    link_6 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link6.stl")
    link_7 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link7.stl")
    hand = load_mesh(f"{robot.config_path}/meshes/frankie/frankie_hand.stl")

    hand_pose = sm.SE3(robot.links[-1].ets[0].A())
    link_7_pose = robot.fkine(robot.q, start=robot.links[-1], include_base=False)
    # for ets in robot.links[-2].ets:
    #     link_7_pose = sm.SE3(ets.A()) * link_7_pose
    link_7_pose = link_7_pose  # * hand_pose
    # link_7_pose = sm.SE3(robot.links[-2].ets[0].A()) * hand_pose
    link_6_pose = robot.fkine(robot.q, start=robot.links[-2], include_base=False)

    link_6.T = link_6_pose.inv()
    link_7.T = link_7_pose.inv()
    hand.T = hand_pose.inv()

    return [link_6, link_7, hand]

if __name__ == "__main__":
    # test interative pose
    pose = False
    mesh = True

    if pose:
        env = swift.Swift()
        env.launch(realtime=True)

        initial_pose = sm.SE3(0, 1, 0) * sm.SE3.Rx(np.pi / 2)
        pose = interative_pose(env, initial_pose)
        while True:
            env.step()

    # test interative mesh
    if mesh:
        env = swift.Swift()
        env.launch(realtime=True)
        robot = NeuralFrankie("points_8")
        robot.q = robot.qr
        env.add(robot)
        link_6 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link6.stl")
        link_7 = load_mesh(f"{robot.config_path}/meshes/frankie/panda_link7.stl")
        hand = load_mesh(f"{robot.config_path}/meshes/frankie/frankie_hand.stl")
        meshes = [link_6, link_7, hand]

        hand_pose = sm.SE3(robot.links[-1].ets[0].A())
        link_7_pose = robot.fkine(robot.q, start=robot.links[-1])
        # for ets in robot.links[-2].ets:
        #     link_7_pose = sm.SE3(ets.A()) * link_7_pose
        link_7_pose = link_7_pose  # * hand_pose
        # link_7_pose = sm.SE3(robot.links[-2].ets[0].A()) * hand_pose
        link_6_pose = robot.fkine(robot.q, start=robot.links[-2])

        pose_to_mesh = [link_6_pose.inv(), link_7_pose.inv(), hand_pose.inv()]

        mesh = interative_mesh(env, meshes, robot.fkine(robot.q), pose_to_mesh)
        while True:
            env.step()
