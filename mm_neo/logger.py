import spatialmath as sm
import pdb

import pandas as pd

from collections import defaultdict
import os
import time
from neural_robot.neural_frankie import NeuralFrankie
from neural_robot.neural_frankie_omni import NeuralFrankieOmni
import numpy as np
from mm_neo.configs.controllers import NeoConfig
from typing import Tuple, Dict, List

import json

from scipy.integrate import trapezoid, cumulative_trapezoid


class Logger:
    def __init__(self, exp_name: str, config: NeoConfig, n_steps = None):
        self.n_steps = n_steps
        self.initialize(exp_name, config)

    def log(self, data, variable_data={}):
        for k, v in data.items():
            self.add_data(k, v)
        for k, v in variable_data.items():
            self.add_variable_data(k, v)

    def initialize(self, exp_name, config):
        self.data = defaultdict(list)
        self.exp_name = exp_name

        self.folder_name = f"logs/{exp_name}"
        os.makedirs(self.folder_name, exist_ok=True)
        self.info_file = f"{self.folder_name}/info.txt"
        if type(config.lamda_q_vec) == np.ndarray:
            config.lamda_q_vec = config.lamda_q_vec.tolist()

        if type(config.lamda_q) == np.ndarray:
            config.lamda_q = config.lamda_q.tolist()

        if type(config.only_base) == np.ndarray:
            config.only_base = config.only_base.tolist()
        if type(config.max_acceleration_step) == np.ndarray:
            config.max_acceleration_step = config.max_acceleration_step.tolist()

        with open(f"{self.folder_name}/config.json", "w") as f:
            f.write(json.dumps(config.__dict__, indent=4))
        self.variable_data = defaultdict(list)

    def add_variable_data(self, name, values):
        """Add data with variable size

        Parameters
        ----------
        name : str
            key
        values : np.ndarray
            values to store
        """
        self.variable_data[name].append(values.tolist())

    def add_data(self, name, values):
        self.data[name].append(values)

    def save(self, robot="frankie"):
        with open(self.info_file, "w") as f:
            # Write the date
            f.write(f"Date: {time.ctime()}\n")

            for key, value in self.data.items():
                f.write(f"{key}\n")

        with open(f"{self.folder_name}/matrix.json", "w") as f:
            f.write(json.dumps(self.variable_data, indent=4))
        self.data["robot"] = robot
        np.savez(f"{self.folder_name}/data.npz", **self.data)


class SwiftLogger:
    def __init__(self, exp_folder: str):
        self.robot = exp_folder.split("/")[-1].split("_")[0]
        try:
            self.exp_name = exp_folder.split("/")[-2]
        except:
            self.exp_name = exp_folder.split("/")[-1]
        self.data = load_data(exp_folder)
        self.q = self.data["q"]
        # self.base = self.data["fixed_base"]
        self.base = self.data["base"]
        self.index = -1

    def reload(self, exp_folder: str):
        self.data = load_data(exp_folder)
        self.q = self.data["q"]
        self.base = self.data["base"]
        # self.base = self.data["fixed_base"]
        self.index = -1

    def get_state(self, index: int):
        """
        get the state of the robot at a given index
        Parameters
        ----------
        index
            index of the state
        """
        self.index = index
        if index >= len(self.q):
            return self.q[-1], self.base[-1]
        return self.q[index], self.base[index]

    def done(self):
        return self.index >= len(self.q)

    def step(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get the next state of the robot

        Returns
        -------
        Tuple[bool, np.ndarray, np.ndarray]
            done, q, base
        """

        self.index += 1
        if self.index >= len(self.q):
            return True, self.q[-1], self.base[-1]
        return False, self.q[self.index], self.base[self.index]


def compute_gt_distance(info, sdf_model, gt_robot):
    # gt_robot = NeuralFrankie("points_0", spheres = False)
    # env = swift.Swift()
    # env.launch(headless=True)
    # env.add(gt_robot)
    joints = get_joints(info)
    distance = get_distance(info)
    bases = get_base(info)
    gt_distances = []
    distance_error = []
    for ix in range(joints.shape[0]):
        gt_robot.q = joints[ix]
        gt_robot.base = bases[ix]
        # env.step()
        X_WSp = gt_robot.transform_points()
        gt_distance = sdf_model.get_distance(X_WSp)
        min_gt_distance = gt_distance.min()
        gt_distance = gt_robot.get_distance_links(gt_distance)
        error = {k: (gt_distance[k] - distance[ix][k]) for k in gt_distance.keys()}
        gt_distances.append(gt_distance)

        distance_error.append(error)
        if min_gt_distance < 0:
            # print("collision")
            break
    return gt_distances, distance_error


def tracking_error(data):
    q = data["q"]
    error = []
    for i in q:
        Te = robot.fkine(i)
        Tep = robot.fkine(robot.qr)
        T_eEp = Tep.inv() * Te
        # spatial error
        et = np.sum(np.abs(T_eEp.t))
        error.append(et)
    return error


def total_translation(data, robot):
    positions = get_eef_traj(data, robot)
    total = 0
    for i in range(len(positions) - 1):
        total += np.linalg.norm(positions[i] - positions[i + 1])
    return total


def get_eef_traj(data, robot):
    q = data["q"]
    base = data["base"]

    positions = []

    for i in range(len(q)):
        robot.base = sm.SE3(base[i], check=False)
        robot.base = robot.base.norm()
        T_We = robot.fkine(q[i])
        xyz = T_We.t
        positions.append(xyz)
    return positions


def get_fk(data):
    robot = NeuralFrankie(spheres=False)
    # robot = NeuralFrankieOmni(spheres = False)
    q = data["q"]
    base = data["base"]

    x = []
    y = []
    z = []
    for i in range(len(q)):
        robot.base = sm.SE3(base[i])
        T_We = robot.fkine(q[i])
        xyz = T_We.t
        x.append(xyz[0])
        y.append(xyz[1])
        z.append(xyz[2])
    return x, y, z


def get_base(data):
    xyz = data["base"]
    base = []
    for p in xyz:
        try:
            base.append(sm.SE3(p, check=False).norm())
        except:
            import pdb

            pdb.set_trace()
    return base


def get_distance(data):
    return data["distance"]


def real_collided(data):
    d = data["gt_distance"]
    for i in d:
        if min(i.values()) < 0:
            return True
    return False


def collided(data):
    d = data["distance"]
    for i in d:
        if min(i.values()) < 0:
            return True
    return False


def reached_real_world(data, threshold=0.021):
    error = data["error"]
    error = np.min(error)
    if error <= threshold:
        return True
    return False


def reached(data, threshold=0.02):
    error = data["error"]
    if real_collided(data):
        return False
    if error[-1] <= threshold:
        return True
    return False


def error_gt(info):
    error_distance = []
    for i in info:
        distances = np.array([*i.values()])
        error_distance.append(min(abs(distances)))
    return error_distance


def get_min_distance(d: List[Dict]):
    # list[dict {joint_name, distance}]
    min_distance = []
    for i in d:
        min_distance.append(min(i.values()))
    return min_distance


def get_min_distance_dict(data):
    min_distance = []
    for i in data:
        min_distance.append(min(i.values()))
    return min_distance


def get_gt_distance(data):
    # gt = data["gt_distance"][0,:]
    # error = data["gt_distance"][1,:]
    gt = data["gt_distance"]
    return gt


def load_data(exp_name: str, relative=True):
    if "logs" not in exp_name:
        exp_name = f"logs/{exp_name}"
    if relative:
        data = np.load(f"{exp_name}/data.npz", allow_pickle=True)
    else:
        data = np.load(f"{exp_name}/data.npz", allow_pickle=True)

    return data


def convert_pandas(data):
    df = pd.DataFrame.from_dict(
        {item: data[item] for item in data.files}, orient="index"
    )
    return df


def load_pandas(exp_name: str):
    data = load_data(exp_name)
    df = pd.DataFrame.from_dict(
        {item: data[item] for item in data.files}, orient="index"
    )
    return df


def load_post(exp_name: str, relative=True):
    if relative:
        data = np.load(f"logs/{exp_name}/data_post.npz", allow_pickle=True)
    else:
        data = np.load(f"{exp_name}/data_post.npz", allow_pickle=True)
    return data


def save_data(exp_name: str, data: dict):
    np.savez(f"logs/{exp_name}/data_post.npz", **data)
    return


def get_joints(data):
    return data["q"]


def get_vel(data):
    return data["qd"]


def smoothness(data):
    abs_qdd = np.abs(np.diff(data["qd"], axis=0))
    smoothness = np.sum(abs_qdd, axis=1)
    cum_smoothness = np.cumsum(abs_qdd, axis=0).sum(axis=1)
    return smoothness, cum_smoothness


def jerkiness(data):
    jerk = np.abs(np.diff(np.diff(data["qd"], axis=0), axis=0))
    Jerk = np.sum(jerk, axis=1)
    cumJerk = np.cumsum(jerk, axis=0).sum(axis=1)
    return Jerk, cumJerk


def central_diff(vel: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = np.zeros_like(vel)
    dt = np.diff(t, axis=0)
    a[0] = (vel[1] - vel[0]) / dt[0]
    a[-1] = (vel[-1] - vel[-2]) / dt[-1]
    for i in range(1, len(vel) - 1):
        a[i] = (vel[i + 1] - vel[i - 1]) / (t[i + 1] - t[i - 1])
    return a


def acc_eef_real_robot(data, robot) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute the acceleration of the end effector
    by computing the rate of change on the eef velocity
    """
    # import matplotlib.pyplot as plt

    qd = data["qd"]
    # qd = data["joint_vel_desired"]
    q = data["q"]
    eef_vels = []
    eef_accs = []
    t = data["time"]

    qdd = central_diff(qd, t)
    for i in range(len(q)):
        robot.q = q[i]
        robot.base = sm.SE3(data["base"][i])
        J = robot.jacob0(q[i])
        J_dot = robot.jacob0_dot(q[i], qd[i], J)
        eef_vel = J @ qd[i]
        eef_acc = J @ qdd[i] + J_dot @ qd[i]
        eef_vels.append(eef_vel)
        eef_accs.append(eef_acc)

    t = t[: len(eef_accs)]
    acc = np.array(eef_accs)[:, :3]
    acc = np.linalg.norm(acc, axis=1)
    cumAcc = cumulative_trapezoid(acc, t)
    return acc, cumAcc


def acc_eef(data, robot, dt=0.05):
    qd = data["qd"]
    q = data["q"]
    eef_accs = []

    t = np.arange(len(qd)) * dt
    qdd = central_diff(qd, t)

    for i in range(len(q)):
        robot.q = q[i]
        robot.base = sm.SE3(data["base"][i], check=False).norm()
        J = robot.jacob0(q[i])
        J_dot = robot.jacob0_dot(q[i], qd[i], J)
        eef_acc = J @ qdd[i] + J_dot @ qd[i]
        eef_accs.append(eef_acc)

    eef_accs = np.array(eef_accs)
    accs = eef_accs[:, :3]  # Remove angular velocity

    acc = np.linalg.norm(accs, axis=1)

    cumAcc = cumulative_trapezoid(acc, t)

    return acc, cumAcc


def jerk_eef(data, robot, t=0.05):
    eef_positions = get_eef_traj(data, robot)
    if t is None:
        dt = 0.05
    elif isinstance(t, np.ndarray):
        dt = np.diff(t, axis=0)[: len(eef_positions)]
        dt = np.mean(dt)
    else:
        dt = t
    eef_vel = np.diff(eef_positions, axis=0) * 1 / dt
    eef_acc = np.diff(eef_vel, axis=0) * 1 / dt
    jerk = np.diff(eef_acc, axis=0) * 1 / dt
    jerk = np.abs(jerk)
    jerk = np.sum(jerk, axis=1)

    cumJerk = np.cumsum(jerk, axis=0)
    return jerk, cumJerk


def acc(data):
    acc = np.abs(np.diff(data["qd"], axis=0))
    Acc = np.sum(acc, axis=1)
    cumacc = np.cumsum(Acc, axis=0).sum(axis=1)
    return Acc, cumacc


def manipulability(data):
    # robot = NeuralFrankie(spheres = False)
    robot = NeuralFrankieOmni(spheres=False)
    q = data["q"]
    mani = []
    for i in q:
        mani.append(
            robot.manipulability(
                i, end=robot.grippers[0], start=robot.links[robot.base_dofs + 2]
            )
        )
    return mani


def load_folder(folder):
    files = os.listdir(folder)
    data = []
    for i in files:
        if os.path.isdir(f"{folder}/{i}"):
            data.append(f"{folder}/{i}")
    return data
