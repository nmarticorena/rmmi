'''
@file: post_process.py

This script is used to post-process the data from the reaching task. to obtain 
the gt distance by using a heavily sampled robot
'''

from neural_robot.unity_frankie import NeuralFrankie
from dataclasses import dataclass, field
import mm_neo.logger as logger
from mm_neo.ideal_sdf import load_from_json
import tyro
import os
from typing import List
from tqdm import tqdm
import numpy as np
@dataclass
class Config():
    envs: List = field(default_factory=lambda: ["bookshelf","bookshelf_2","table_free" ,"table","table_cyl_small"])
    exp_name: str = "logs/run_all_2/spheres"
    
args = tyro.cli(Config)

gt_robot = NeuralFrankie("points_0", spheres = False)
def process_var(env, var):
    instance = os.listdir(os.path.join(args.exp_name, env, var))
    for run in tqdm(instance, position = 2 , leave = False):
        run_path = os.path.join(args.exp_name, env, var, run)
        data = logger.load_data(run_path, relative=False) 
        gt_distance, error = logger.compute_gt_distance(data, sdf_model, gt_robot)
        # import pdb; pdb.set_trace()

        new_data = {**data}
        new_data["gt_distance"] = gt_distance
        new_data["error_distance"] = error
        # logger.save_data(run_path, new_data)
        np.savez(f"{run_path}/data_post.npz", **new_data)
     
# envs = os.listdir(args.exp_name)



envs = args.envs
for env in tqdm(envs, position = 0):
    sdf_model,_ ,_ = load_from_json(f"/media/nmarticorena/DATA/ideal_sdfs/{env}.json")
    variations = os.listdir(os.path.join(args.exp_name, env))
    for var in tqdm(variations, position = 1, leave = False):
        process_var(env, var)
 
