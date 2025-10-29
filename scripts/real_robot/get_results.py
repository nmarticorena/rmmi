import os
import pdb
import matplotlib.pyplot as plt

from numpy.core.numeric import ones_like
from torch import exp
import mm_neo.logger as logger
import pandas as pd
import numpy as np
from neural_robot.neural_frankie import NeuralFrankie
from dataclasses import dataclass, field
import tyro
from typing import List

from nerf_tools import plots

plots.configure_matplotlib()
np.set_printoptions(precision=2, suppress=True)


@dataclass
class Config:
    exp_name: str = "test_random/"
    num_exp: int = 10
    plot: bool = False


exp_name = "test_s12"
args = tyro.cli(Config)

if "logs" not in args.exp_name:
    experiment_name = "logs/" + args.exp_name
else:
    experiment_name = args.exp_name
folders = os.listdir(experiment_name)
try:
    folders.remove("results")
    folders.remove("latex_tables")
except Exception as e:
    print(e)
    pass
os.makedirs(f"{experiment_name}/results", exist_ok=True)
os.makedirs(f"{experiment_name}/latex_tables", exist_ok=True)


robot = NeuralFrankie("points_9", spheres=False)

df = pd.DataFrame()
for i in range(args.num_exp):
    data = logger.load_data(f"{experiment_name}/exp_{i}", relative=False)
    odom_data = logger.load_data(f"{experiment_name}/exp_{i}_odom", relative=False)
    fixed_data = logger.load_data(f"{experiment_name}/exp_{i}_fixed", relative=False)
    eef_acc_odom, eef_cum_acc_odom = logger.acc_eef_real_robot(odom_data, robot)

    time = data["time"]
    if args.plot:
        plt.cla()
        plt.plot(
            time[: len(data["qd"])] - time[0],
            data["qd"],
            label=[f"qd_{i}" for i in range(9)],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint vel [rad]")
        plt.title("Joint velocities [rad/s] at controller rate")
        plt.savefig(f"{experiment_name}/results/joint_velocities_{i}.png")

        qdd = logger.central_diff(odom_data["qd"], time)
        plt.cla()
        plt.plot(
            time[: len(qdd)] - time[0],
            qdd,
            label=[f"qqd_{i}" for i in range(9)],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint vel [rad]")
        plt.title(r"Joint acceleration $[rad/s^{2}]$ at odometry rate")
        plt.savefig(f"{experiment_name}/results/joint_acc_{i}.png")

        # Plot end effector acceleration
        plt.cla()
        plt.plot(
            time[: len(eef_acc_odom)] - time[0],
            eef_acc_odom,
            label=[r" $a_{e}$$[m/s^{2}]$"],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel(r"End effector acceleration $[m/s^{2}]$")
        plt.title("End effector acceleration at odometry rate")
        plt.savefig(f"{experiment_name}/results/eef_acc_{i}.png")

        plt.cla()
        plt.plot(
            time[: len(data["qd"])] - time[0],
            data["joint_vel_desired"],
            label=[f"qd_{i}" for i in range(9)],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint vel [rad]")
        plt.title("Joint desired [rad/s] at controller rate")
        plt.savefig(f"{experiment_name}/results/joint_desired_{i}.png")

        plt.cla()
        plt.plot(
            odom_data["time"][: len(odom_data["qd"])] - time[0],
            odom_data["qd"],
            label=[f"qd_{i}" for i in range(9)],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint vel [rad]")
        plt.title("Joint velocities [rad/s] at odom rate")
        plt.savefig(f"{experiment_name}/results/joint_velocities_odom_{i}.png")

        plt.cla()
        plt.plot(
            fixed_data["time"][: len(fixed_data["qd"])] - time[0],
            fixed_data["qd"],
            label=[f"qd_{i}" for i in range(9)],
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint vel [rad]")
        plt.title("Joint velocities [rad/s] at fixed rate 10Hz")
        plt.savefig(f"{experiment_name}/results/joint_velocities_fixed_{i}.png")
    odom_time = odom_data["time"]

    plot_time = odom_data

    dt = np.diff(plot_time["time"], axis=0)[: len(plot_time["qd"])]
    dt = np.insert(dt, 0, dt[0])
    # eef_acc, eef_cum_acc = logger.acc_eef_real_robot(data, robot)
    # eef_acc_fixed, eef_cum_acc_fixed = logger.acc_eef_real_robot(fixed_data, robot)

    distance = data["error"]
    closest = np.min(distance, axis=0)
    reached = logger.reached_real_world(data, 0.02)

    # real_collided = logger.real_collided(data)
    values = {
        "name": f"exp_{i}",
        "final end effector distance": closest,
        "reached": reached,
        "mean eef with odom": np.mean(eef_acc_odom),
        "eef acc odom": eef_cum_acc_odom[-1],
        # "eef mean with fixed": np.mean(eef_acc_fixed),
        # "eef acc fixed": eef_cum_acc_fixed[-1],
        # "eef mean with u": np.mean(eef_acc),
        # "eef acc u": eef_cum_acc[-1],
    }
    df_temp = pd.DataFrame([values])

    df = pd.concat([df, df_temp], ignore_index=True)
    data.close()

# exit(1)
results = df.drop("name", axis=1)
# add new column with averages
results.loc["average"] = results.mean(axis=0)

# pdb.set_trace()
with open(f"{experiment_name}/results/env_name_result.txt", "w") as f:
    f.write(results.to_string())
results.to_excel(f"{experiment_name}/results/results.xlsx")
print(results)
