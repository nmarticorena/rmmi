import os
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


@dataclass
class Config:
    exp_name: str = "test_random/"
    num_exp: int = 10
    spheres: bool = False
    no_active: bool = False


args = tyro.cli(Config)
if args.no_active:
    folder_template = "logs/" + args.exp_name + "_var_{}_no_active"
elif args.spheres:
    folder_template = "logs/" + args.exp_name + "_var_{}_spheres"
else:
    folder_template = "logs/" + args.exp_name + "_var_{}"

experiment_name = folder_template.format(1)
folders = os.listdir()
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

for j in range(1, 5):
    for i in range(args.num_exp):
        data = logger.load_data(folder_template.format(j) + f"/exp_{i}", relative=False)
        odom_data = logger.load_data(
            folder_template.format(j) + f"/exp_{i}_fixed", relative=False
        )
        time = data["time"]
        q = data["q"]
        eef_acc_odom, eef_cum_acc_odom = logger.acc_eef_real_robot(
            odom_data, robot, 1 / 10
        )
        distance = data["error"]
        closest = np.min(distance, axis=0)
        reached = logger.reached_real_world(data, 0.02)
        values = {
            "name": f"exp_{i}",
            "final end effector distance": closest,
            "reached": reached,
            "mean eef with pos": np.mean(eef_acc_odom),
            "eef acc pos": eef_cum_acc_odom[-1],
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
