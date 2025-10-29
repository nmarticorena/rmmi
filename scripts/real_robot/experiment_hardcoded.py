import rospy

import spatialmath as sm
import numpy as np

from mm_neo.ros_controllers import CollisionAvoidance as RosInterface
from mm_neo.configs.controllers import NeoConfig, Status
from mm_neo.sdf import load_sdf_swift
from mm_neo.logger import Logger

from std_msgs.msg import String

import json
from dataclasses import dataclass
import tyro
from datetime import datetime
from neural_robot.utils import points_utils as pu


arm_home = np.array([0.0, 0.0, 0.131, -1.43, -0, -2.796, -0.023, 1.42, 0.856])


np.random.seed(125)


def save():
    global logger
    global args
    global Controller
    Controller.stop()


@dataclass
class Args:
    env_name: str = "boxes"
    env_type: str = "ARKit"
    exp_name: str = "s_12"
    controller: NeoConfig = NeoConfig()
    log: bool = False
    trials: int = 5  # number of times each pose
    loop: bool = False
    prepose: bool = True
    spheres: bool = False


args = tyro.cli(Args)
args.controller.log = args.log
args.controller.approx_jacobian = False
args.controller.real_robot = True
rospy.init_node("mm_neo_experiment_real")

if "table" in args.env_name:
    prepose = sm.SE3(1.2, 0, 0.2)
elif "cabinet" in args.env_name:
    prepose = sm.SE3(0, -1, 0)
else:
    prepose = sm.SE3(0, -0.5, 0)

logger = None

mesh, sdf = load_sdf_swift(args.env_name, args.env_type, sm.SE3())
with open(f"./data/isdf_poses/{args.env_name}.json", "r") as f:
    config = json.load(f)
    poses = [sm.SE3(np.array(pose), check=False) for pose in config["poses"]]
    poses = [pose.norm() for pose in poses]
    base = sm.SE3(np.array(config["base"]), check=False).norm() * sm.SE3.Rz(np.pi / 2)
    print(poses)
if args.spheres:
    robot_name = "curobo_final"
    args.controller.control_frecuency = 120
    args.controller.ds = float(pu.get_min_distance("curobo")) * 1.1
    # args.controller.ds = 0.0
else:
    robot_name = "real_robot_2"
    args.controller.control_frecuency = 75
    args.controller.ds = (
        float(pu.get_min_distance(robot_name.replace("post_", ""))) * 1.1
    )  # extra 10 % for considering the rror from isdf


Controller = RosInterface(
    args.controller,
    poses,
    sdf,
    mesh.filename,
    logger=logger,
    robot_name=robot_name,
    spheres=args.spheres,
    max_attempts=15 * args.controller.control_frecuency,
)
done = False

rospy.on_shutdown(save)

Controller.set_status(Status.arm, 0)
while not done:
    done, Controller.qd = Controller.vel_controller.step(arm_home)
    Controller.qd *= 0.35
    Controller.step()
Controller.set_status(Status.base, 0)
nav = Controller.base.send_goal(base)

Controller.step_pose()

done = False


indexes = np.repeat(np.arange(len(poses)), args.trials)
print(indexes)
np.random.shuffle(indexes)
# for i in range(18):
#     random_y = np.random.uniform(-0.2, 0.2)
#

# indexes = [indexes[18]]
done_pose = False
trials = 0

Controller.set_status(Status.running, trials)
Controller.select_target(indexes[trials])
Controller.select_target(indexes[trials])
main_message = ""
while not rospy.is_shutdown():
    done, message = Controller.step_pose()
    if done:
        print(message)
        if done_pose:
            trials += 1
            done_pose = False
            print("reset")
            if args.loop:  # and main_message != "Timeout":  # Reset if fails one
                Controller.set_status(Status.arm, trials)
                Controller.stop()
                rospy.sleep(0.1)
                if trials >= args.trials * len(poses):
                    break
                index = indexes[trials]
                Controller.select_target(index)
                Controller.set_status(1, trials)
                Controller.select_target(index)
            else:
                home = False
                Controller.set_status(3, trials)
                rospy.sleep(0.1)
                while not home:
                    home, Controller.qd = Controller.vel_controller.step(arm_home)
                    Controller.qd *= 0.35
                    Controller.step()
                random_y = np.random.uniform(-0.2, 0.2)
                Controller.set_status(2, trials)
                rospy.sleep(0.1)
                Controller.base.send_goal(base * sm.SE3.Ty(random_y))
                Controller.set_status(1, trials)
                rospy.sleep(0.1)
            print(trials)
            if trials >= args.trials * len(poses):
                break
            index = indexes[trials]
            Controller.select_target(index)

        else:
            if args.prepose:
                Controller.set_status(4, trials)
                rospy.sleep(0.1)
                if Controller.target_pose.t[2] < 0.45:
                    print("to low")
                    Controller.pre_pose(sm.SE3(0, 0, 0.5))

                if message == "Timeout":
                    print("Going to reset the arm")
                    if "table" in args.env_name:
                        Controller.pre_pose(sm.SE3(1, 0, 0))

                    else:
                        # Controller.pre_pose(sm.SE3(0, -0.25, 0))
                        Controller.pre_pose(prepose)
                else:
                    Controller.pre_pose(prepose)
                main_message = message
                done_pose = True
            else:
                done_pose = True

    else:
        Controller.step()

# Controller.pre_pose(prepose)
# done, message = Controller.step_pose()
#
# while not done:
#     done, message = Controller.step_pose()
#     Controller.step()
#
#
Controller.set_status(0, trials + 1)
print("Done")
