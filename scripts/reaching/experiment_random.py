import numpy as np
import traceback
import spatialmath as sm
import torch
from tqdm import tqdm

# import time
import swift
from mm_neo.ideal_sdf import RandomizedSDF, RandomizedSDFMultiple
import os
from mm_neo.configs.controllers import NeoConfig
from mm_neo.logger import Logger
from mm_neo.sdf import load_isdf
from mm_neo.mm_controllers import MMNeoSDF

from neural_robot.unity_frankie import NeuralFrankie
import tyro
from dataclasses import dataclass, field

from neural_robot.utils.points_utils import get_min_distance

np.random.seed(125)  # set equal seed for the different poses
torch.manual_seed(125)


# Translator dict
names = {"table_new": "table_new_v2",
         "bookshelf_cage": "bookshelf_cage_v2"}


@dataclass
class Config:
    n_experiments: int = 1000
    env_name: str = "random_table"
    exp_name: str = "test_random/"
    config: NeoConfig = field(default_factory = lambda: NeoConfig())
    headless: bool = True
    realtime: bool = False
    spheres: bool = False
    robot: str = "points_1"
    multiple_poses: bool = False


args = tyro.cli(Config)

args.exp_name = f"random_reaching/{args.exp_name}"

dt = 0.05


config = args.config
# config.acceleration_gain = 10000000
# config.vel_scaler = 1
env = swift.Swift()
env.launch(realtime=args.realtime, headless=args.headless, browser="firefox")
# if args.env_name == "table_new":
#     env.set_camera_pose([1.75, 0.5, 1.25], [1, 0, 0.5])
# elif args.env_name == "bookshelf_cage":
#     print("here")
# env.set_camera_pose([-0.1, 0.5, 1.75], [5.0, 0.0, 0.15])


if args.spheres:
    robot_sampled = "curobo"
    robot = NeuralFrankie(f"{robot_sampled}", spheres=True)
    # args.config.ds = get_min_distance(robot_sampled) * 1.25
    print(f"Using {args.config.ds} as stopping distance")
else:
    robot_sampled = args.robot
    args.config.ds = get_min_distance(args.robot, "compare_link_6") * 1.1
    robot = NeuralFrankie(f"{robot_sampled}", spheres=False)

gt_robot = NeuralFrankie("points_0", spheres=False)
# gt_robot = None
env.add(gt_robot)

robot.q = robot.qr

if not args.headless:
    robot.replace_point_meshes()
env.add(robot)

# sampled = os.listdir(f"data/{args.env_name}")
# sampled = [s for s in sampled if "table" in s]

# if len(sampled) > args.n_experiments:
    # sampled = sampled[: args.n_experiments]

if args.multiple_poses:
    problem = RandomizedSDFMultiple(
        f"data/{args.env_name}/generator.json", env, headless=args.headless
    )
else:
    problem = RandomizedSDF(
        f"data/{args.env_name}/generator.json", env, headless=args.headless
    )


robot_type = type(robot).__name__

logger = Logger(args.exp_name, config)
reached = False

# objective = problem.target
for i in tqdm(range(args.n_experiments), leave=False):
    filename = f"data/{args.env_name}/{args.env_name}_{i:04d}.json"
    problem.load(filename)
    col = "" if config.collisions else "_no_col"
    if config.collision_cost == "":
        col += "_no_active_col"
    approx = "approx" if config.approx_jacobian else "exact"
    file_name = filename.split(".")[0]

    folder = f"{args.exp_name}/{args.env_name}/"  # template of folder
    folder += f"{col}_{config.collision_cost}_{approx}/"  # Config
    folder += f"{robot_type}_{robot_sampled}_{file_name}"  # robot
    isdf_folder =f"{names[args.env_name]}/{names[args.env_name]}_{i:04d}"

    
    sdf = load_isdf(isdf_folder)

    controller = MMNeoSDF(robot,sdf, config, logger=logger, gt_robot=gt_robot, gt_obstacles=problem.get_obstacles())
    logger.initialize(folder, config)
    robot.base = problem.base
    robot.q = robot.qr
    robot.qd = robot.qz

    gt_robot.base = problem.base
    gt_robot.q = robot.qr
    gt_robot.qd = robot.qz
    if args.multiple_poses:
        original_targets = problem.targets
        targets = []
        for t in original_targets:
            targets.append(t)
            targets.append(t * sm.SE3(-0.3, 0, 0))
    else:
        targets = [problem.target]
    for target in targets:
        for j in range(300):
            try:
                reached, robot.qd[:], failed = controller.step(target)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                failed = True
                break
            if reached:
                break
            if failed:
                print("failed")
                break
            env.step(dt)
            base_new = robot.fkine(robot._q, end=robot.links[robot.base_dofs]).A
            robot._T = base_new
            robot.q[: robot.base_dofs] = 0
            gt_robot._q = robot._q
            gt_robot._T = robot._T
            gt_robot.q[: robot.base_dofs] = 0
        if not reached:
            break
    print(f"experiment number {i} got reached : {reached}")
    reached = False
    logger.save(type(robot).__name__)

env.step()

# time.sleep(1)
