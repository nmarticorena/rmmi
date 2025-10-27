"""
@file: fix_poses.py

This script runs colmap on a nerfcapture dataset, then procced
to obtain the scale between colmap scale and the arkit scale

Finally proceed to save the scaled poses in a new json file
"""

import tyro
from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

import os

INGP = os.environ["INGP"]
NERF_CAPTURE = os.environ["NERF_CAPTURE"]
ROBOT = os.path.join(os.environ["MM"], "results", "real_scans")


@dataclass
class args:
    capture_name: str = "boxes"
    run_colmap: bool = False
    robot: bool = False


Args = tyro.cli(args)

if Args.robot:
    capture_folder = os.path.join(ROBOT, Args.capture_name)
else:
    capture_folder = os.path.join(NERF_CAPTURE, Args.capture_name)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# get camera_params
with open(capture_folder + "/transforms.json") as f:
    dataset = json.load(f)

w = int(dataset["w"])
h = int(dataset["h"])
fl_x = dataset["fl_x"]
fl_y = dataset["fl_y"]
cx = dataset["cx"]
cy = dataset["cy"]

# get ate.txt of arkit
with open(capture_folder + "/arkit.txt", "w") as f:
    for frame in dataset["frames"]:
        camera_index = frame["file_path"].split("/")[-1].split(".")[0]
        pose = np.array(frame["transform_matrix"])
        rot = R.from_matrix(pose[:3, :3])
        qx, qy, qz, qw = rot.as_quat()
        f.write(
            f"{camera_index} {pose[0,3]} {pose[1,3]} {pose[2,3]} {qx} {qy} {qz} {qw}\n"
        )

if Args.run_colmap:
    # # Get features
    if Args.robot:
        os.system(f"colmap feature_extractor --database_path {capture_folder}/database.db \
        --image_path {capture_folder} \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_params {fl_x},{fl_y},{cx},{cy}")
    else:
        os.system(f"colmap feature_extractor --database_path {capture_folder}/database.db \
        --image_path {capture_folder}/images \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_params {fl_x},{fl_y},{cx},{cy}")

    # # get Matches
    os.system(f"colmap exhaustive_matcher --database_path {capture_folder}/database.db")

    # # Run mapping
    if Args.robot:
        os.system(f"colmap mapper --database_path {capture_folder}/database.db \
                --image_path {capture_folder} \
                --output_path {capture_folder}")
    else:
        os.system(f"colmap mapper --database_path {capture_folder}/database.db \
                --image_path {capture_folder}/images \
                --output_path {capture_folder}")

    # Export result to txt
    os.system(f"colmap model_converter --input_path {capture_folder} \
                --output_path {capture_folder} \
                --output_type TXT")


# Load colmap poses
poses = []
poses_ate_format = []

cameras = []

with open(f"{capture_folder}/images.txt", "r") as file:
    lines = file.readlines()

# Skip the first 4 lines
lines = lines[4:]

# Read every second line
info_lines = [line.strip() for i, line in enumerate(lines) if i % 2 == 0]

del dataset["frames"]

dataset["frames"] = []

with open(capture_folder + "/colmap.txt", "w") as f_ate:
    for info in info_lines:
        cam_index = info.split(" ")[-1]
        cam_index = int(cam_index.split(".")[0])
        # Pose is in the first line [tx,ty,tz] [3x3 rot matrix]

        # id, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        values = info.split(" ")[1:8]
        pose = np.eye(4)
        pose[:3, 3] = np.array([float(i) for i in values[-3:]])
        tvec = np.array([float(i) for i in values[-3:]])
        # pose[:3,:3] = np.array([float(i) for i in values[3:]]).reshape((3,3))
        poses.append(pose)
        qvec = np.array(tuple(map(float, values[0:4])))
        tvec = np.array(tuple(map(float, values[4:])))
        # qvec  = values[0:4]
        Ro = qvec2rotmat(-qvec)
        t = tvec.reshape([3, 1])
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        m = np.concatenate([np.concatenate([Ro, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)
        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1  # flip whole world upside down

        print(c2w)
        rot = R.from_matrix(c2w[:3, :3])
        qx, qy, qz, qw = rot.as_quat()
        tx, ty, tz = c2w[:3, 3]

        frame = {
            "file_path": f"images/{cam_index}.png",
            "transform_matrix": c2w.tolist(),
            "depth_path": f"images/{cam_index}.depth.png",
        }

        dataset["frames"].append(frame)
        f_ate.write(f"{cam_index} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


with open(capture_folder + "/colmap.json", "w") as f:
    json.dump(dataset, f, indent=4)


print(poses)
