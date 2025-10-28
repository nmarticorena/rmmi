from dataclasses import dataclass
from typing import Literal
import enum

# import pprint
import numpy as np

# for omron acc_x = 1 mt/s^2 acc_w = 360 deg/s^2
rad = np.pi / 180

max_acceleration = np.array([1, 180 * rad, 15, 7.5, 10, 12.5, 15, 20, 20])


class Status(enum.IntEnum):
    """
    Status of the experiment

    0 - Done
    1 - Running
    2 - Base
    3 - Arm
    4 - Prepose
    """

    done = 0
    "Waiting for the start signal"
    running = 1
    "Going to the target pose"
    base = 2
    "Resetting the base"
    arm = 3
    "Resetting the arm"
    prepose = 4
    "Resetting the pose after the robot reached a target pose"

class Slack(
    enum.IntEnum
):  # an Int enum allows to serialize to json to save it as a config
    "Slack type"

    normal = 0
    fixed_free = 1
    fixed_constraint = 2
    free_rot = 3
    rpy = 4

class Manipulability(enum.IntEnum):
    "Manipulability cost type"

    disable = 0
    active = 1
    opposite = 2

@dataclass
class NeoConfig:
    "Configuration for the NEO controller"

    max_steps: int = 200
    "Max Number of steps to run the experiment"
    step_time: float = 0.05
    " Step time in seconds"
    beta: float = 2.5
    " PBS Gain term"
    di: float = 0.5
    " Influence Distance"
    ds: float = 0.015
    " Stopping distance"
    xi: float = 1.0
    " Gain of collision avoidance"
    lamda_q: float = 0.01
    " Adjust the trade-off between minimising the joint velocity norm compared to maximising manipulability"
    eta: float = 1.0
    " Gain of the joint-limit avoidance term"
    rho_i: int = 50
    " Influence distance for join limit avoidance unt Degree"
    rho_s: int = 10
    " Minimun distance for join limit unit Degree default = 2"
    exploration_type: int = 1
    " [no exploration, planner, retreat heuristic]"
    min_exploration_steps: int = 10
    " Number of steps to retreat"
    precision: float = 0.02
    " Precision of the end error"
    collisions: bool = True
    " Use collision avoidance"
    waypoint_threshold: float = 0.05
    " Distance to consider that the arm reached a waypoint"
    log: bool = True
    " Log the data "
    rviz: bool = False
    " Publish ros info to debug any error"
    collision_cost: Literal["", "min", "avg", "w_avg", "w2_avg", "w3_avg"] = "w_avg"
    " Use collision cost in addition to the manipulabity as linear cost function"
    gt_collisions: bool = False
    "Use pybullet for compute the collisions"
    orientation_cost: bool = True
    "Use orientation of the base w/r to the gripper"
    collision_gain: float = 1.0
    " Gain of the collision cost lamda_c^{max}"
    approx_jacobian: bool = False
    " Use the approximated jacobian"
    manipulability: Literal[
        Manipulability.disable, Manipulability.active, Manipulability.opposite
    ] = Manipulability.active
    " Use manipulability as cost function "
    home_cost: bool = False
    " Use cost function to return to home position"
    n_dof: int = 9
    "Number of degrees of freedom"
    fixed_slack: Literal[
        Slack.normal,
        Slack.fixed_free,
        Slack.fixed_constraint,
        Slack.free_rot,
        Slack.rpy,
    ] = Slack.normal
    " Use fixed slack , This allows to debug is the auxiliar tasks are formulated correctly"
    fixed_slack_value_free: float = 1e-10
    " Value of the fixed slack for the free joints"
    fixed_slack_value_constraint: float = 1e9
    "Value of the fixed slack for the constraint joints"
    only_min: bool = False
    "Only use the minimum distance for each joint in the collision constrain in a topk"
    topk: int = 25
    "Number of topk to use in the collision constrain"
    max_ev: float = 0.5
    "max spatial velocity of the end effector"
    acc_constraint: bool = False
    "Use acceleration constraint"
    acceleration_gain: float = 1.0
    "Gain for the acceleration limit"
    vel_scaler: float = 0.5
    "Max acceleration for each joint"
    real_robot: bool = False
    "Flag for adding the acceleration limit for the real robot"
    control_frecuency: float = 100.0
    "Control frecuency in Hz"

    def __post_init__(self):
        global max_acceleration
        self.ps = np.radians(self.rho_s)
        self.pi = np.radians(self.rho_i)
        # self.ps = 0.05
        # self.pi = 0.9
        only_base = [1e10, 1e10]
        self.lamda_q_vec = self.lamda_q * np.ones(self.n_dof)
        only_base.extend(self.lamda_q_vec[2:])
        self.max_acceleration_step = (
            max_acceleration * self.acceleration_gain / self.control_frecuency
        )
        self.only_base = np.array(only_base)

if __name__ == "__main__":
    import tyro

    tyro.cli(NeoConfig)
