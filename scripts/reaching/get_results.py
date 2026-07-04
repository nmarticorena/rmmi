import os
import pdb
import mm_neo.logger as logger
import pandas as pd
import numpy as np
from neural_robot.neural_frankie import NeuralFrankie
from dataclasses import dataclass, field
import tyro
import sys
from typing import List


@dataclass
class Config:
    envs: List = field(
        default_factory=lambda: [
            "bookshelf",
            "bookshelf_2",
            "table_free",
            "table",
            "table_cyl_small",
        ]
    )
    exp_name: str = "test_random/"


args = tyro.cli(Config)
experiment_name = "logs/" + args.exp_name
# print(experiment_name)

master_folder = args.envs
master_folder = [experiment_name + "/" + i for i in master_folder]

folders = os.listdir(master_folder[0])
# print(folders)
# exit(0)
os.makedirs(f"{experiment_name}/results", exist_ok=True)
os.makedirs(f"{experiment_name}/latex_tables", exist_ok=True)


robot = NeuralFrankie("points_9", spheres=False)

# gt = False

for m in master_folder:
    env_name = m.split("/")[-1]
    env_df = pd.DataFrame()
    env_successfull_df = pd.DataFrame()
    for j in folders:
        try:
            data_path = logger.load_folder(m + "/" + j)
        except:
            continue
        df = pd.DataFrame()
        df_2 = pd.DataFrame()
        for i in data_path:
            try:
                data = logger.load_post(i, relative=False)
            except Exception as e:
                data = logger.load_data(i, relative=False)

            try:
                collided = logger.collided(data)
            except:
                collided = False
            if len(data["q"]) == 1:
                print(f"Skipping {i} because it has only one pose")
                continue
            eef_acc, eef_cum_acc = logger.acc_eef(data, robot)
            eef_jerk, eef_cum_jerk = logger.jerk_eef(data, robot)
            reached = logger.reached(data)

            real_collided = logger.real_collided(data)
            values = {
                "name": i,
                "real collided": real_collided,
                "reached": reached,
                "eef acc": eef_cum_acc[-1],
                "max eef acc": np.max(eef_acc),
                "mean eef acc": np.mean(eef_acc),
                "min eef acc": np.min(eef_acc),
                "eef jerk": eef_cum_jerk[-1],
                "max eef jerk": np.max(eef_jerk),
                "mean eef jerk": np.mean(eef_jerk),
            }
            # Create a DataFrame object
            df_temp = pd.DataFrame([values])
            if values["reached"]:
                df_2 = pd.concat([df_2, df_temp], ignore_index=True)

            # concatenate the dataframes
            df = pd.concat([df, df_temp], ignore_index=True)
            data.close()
        results = df.drop("name", axis=1).mean().T
        try:
            filter_results = df_2.drop("name", axis=1).mean().T
            env_successfull_df = pd.concat(
                [env_successfull_df, filter_results.rename(f"{j.replace('_', ' ')}")],
                axis=1,
            )

        except KeyError as e:
            print(e)

        with open(f"{experiment_name}/results/env_name_result.txt", "w") as f:
            f.write(results.to_string())
        env_df = pd.concat([env_df, results.rename(f"{j.replace('_', ' ')}")], axis=1)
        df.to_excel(f"{experiment_name}/results/{env_name}_{j}.xlsx")

    env_successfull_df.to_excel(
        f"{experiment_name}/results/{env_name}_successfull.xlsx"
    )
    env_successfull_df.to_csv(f"{experiment_name}/results/{env_name}_successfull.csv")
    env_df.to_excel(f"{experiment_name}/results/{env_name}.xlsx")
    env_df.to_csv(f"{experiment_name}/results/{env_name}.csv")
    env_df.to_latex(
        f"{experiment_name}/latex_tables/{env_name}.tex",
        float_format="%.3f",
        formatters={"name": str.upper},
    )
    env_successfull_df.to_latex(
        f"{experiment_name}/latex_tables/{env_name}_successfull.tex",
        float_format="%.3f",
        formatters={"name": str.upper},
    )
