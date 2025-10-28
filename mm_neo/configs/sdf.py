from dataclasses import dataclass
import os
from typing import Literal, Union, Optional
import tyro
import mm_neo

mm_neo_path = os.path.dirname(mm_neo.__file__) + "/../"


def _get_env_availables() -> list[str]:  # -> list:
    envs = []
    for files in os.listdir(f"{mm_neo_path}/configs/"):
        if files.endswith(".json"):
            envs.append(files[:-5])
    return envs


@dataclass
class iSDF_args:
    env_name: str = "boxes"
    env_type: str = "ARKit"
    folder: str = f"{mm_neo_path}/results/iSDF/{env_name}/0"


@dataclass
class DummySDFArgs:
    env_name: Literal[tuple(_get_env_availables())] = "kitchen_1"
    folder: str = f"{mm_neo_path}/data/kitchen/"


@dataclass
class DummySDFDataset:
    folder_name: str = f"{os.environ['HOME']}/Documents/tools/pybullet_kitchen/data"
    exp_name: str = "kitchen_1"


def get_folder_name(sdf_type: Literal["iSDF", "DummySDF"]) -> str:
    if sdf_type == "iSDF":
        return iSDF_args().folder
    elif sdf_type == "DummySDF":
        return DummySDFArgs().folder


@dataclass
class idealSDF:
    env_name: str = "bookshelf"
    variable: bool = False
    multiple_poses: bool = False


@dataclass
class MeshArgs:
    env_name: str = "kitchen_1"
    sdf_type: Literal["iSDF", "DummySDF"] = "DummySDF"
    folder: Optional[str] = None 
    env_type: str = "ARKit"
    ros: bool = False

    def __post_init__(self):
        if self.folder is None:
            self.folder = get_folder_name(self.sdf_type)

#
# @dataclass
# class sdf_type:
#     sdf_type: iSDF_args = iSDF_args()


@dataclass
class VSConfig:
    """
    Configuration for the vertical scanner
    """

    w: float = 0.3  # Width
    h: float = 0.1  # Height
    n: int = 10  # Number of steps horizontally
    m: int = 3  # Number of steps vertically
    x_offset: float = 0.75  # Offset in x direction
    z_offset: float = 1.15  # Offset in z direction
    min_angle_x: float = 30  # Minimum angle to scan in degrees
    min_angle_y: float = -10  # Minimum angle to scan in degrees
    max_angle_y: float = -40  # Maximum angle to scan in degrees

    # min_angle_x: float = 30 # Minimum angle to scan in degrees
    # min_angle_y: float = -10 # Minimum angle to scan in degrees
    # max_angle_y: float = -20 # Maximum angle to scan in degrees
