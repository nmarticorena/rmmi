import pdb
from neural_robot.unity_frankie import NeuralFrankie
import swift
from mm_neo.ideal_sdf import load_from_json
import tyro
from dataclasses import dataclass
import mm_neo.logger as logger
import os
import time
from mm_neo.utils.swift_utils import set_camera_robot
from mm_neo.sdf import load_sdf_swift
import spatialmath as sm


@dataclass
class Config:
    folder: str = "test"
    exact: bool = True
    env_name: str = "bookshelf_cupboard_flat"


play_state = False

args = tyro.cli(Config)

folder = args.folder

mesh, sdf = load_sdf_swift(args.env_name, "ARKit", sm.SE3())
env = swift.Swift()
env.launch(headless=False, realtime=False, browser="chromium")
env.add(mesh)


robot = NeuralFrankie

variations = os.listdir("logs/" + folder + "/")
if args.exact:
    variations = [i for i in variations if "exact" in i]
else:
    variations = [i for i in variations if "exact" not in i]


loggers = [logger.SwiftLogger(f"{folder}")]
robots = [robot(f"points_1")]


def update_robots(step):
    for i in range(len(robots)):
        robots[i].q, robots[i].base = loggers[i].get_state(step)
    env.step()


def update_exp(step):
    for i in loggers:
        i.reload(folder + f"/{i.exp_name}/{args.robot}_{step}_{robot_sampled}")
    global slider
    update_robots(int(slider.value))


def update_slider(slider, _):
    slider.value = slider.value + 1


def back_slider(slider, _):
    slider.value = slider.value - 1


def play_callback(_):
    global play_state
    play_state = not play_state


play = swift.Button(play_callback, "Play")
env.add(play)

loggers[0].get_state(0)
robots[0].q, robots[0].base = loggers[0].get_state(0)
set_camera_robot(env, robots[0].base, [2, 2, 2])

for robot in robots:
    env.add(robot)

sequence_len = len(loggers[0].q)

slider = swift.Slider(update_robots, 0, len(loggers[0].q), 1, desc="Step")

env.add(slider)
plus_1 = swift.Button(lambda x: update_slider(slider, x), "+1 steps")
minus_1 = swift.Button(lambda x: back_slider(slider, x), "-1 steps")

env.add(plus_1)
env.add(minus_1)


while True:
    if play_state:
        slider.value = (slider.value + 1) % sequence_len
        time.sleep(0.01)
    # env.step(0.0001)
    env.step()
