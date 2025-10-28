from typing import List
import numpy as np
from collections import defaultdict
import spatialgeometry as sg
import spatialmath as sm
import spatialmath.base as smb
import yaml

from curobo.geom.types import Cuboid, WorldConfig , Cylinder
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

def sm_to_curobo_tensor(sm: sm.SE3, tensor_args: TensorDeviceType): 
    """
    Transform a spatial math transformation to a curobo transformation

    Parameters
    ----------
    sm : sm.SE3
        spatial math transformation
    Returns
    -------
    goal_pose: curobo.types.math.Pose
    """
    xyz = sm.t
    quat = smb.r2q(sm.R)
    goal_pose = Pose(
        position=tensor_args.to_device(xyz),
        quaternion=tensor_args.to_device(quat),
    )
    return goal_pose

def sm_to_curobo(sm: sm.SE3):
    """
    Transform a spatial math transformation to a curobo transformation

    Parameters
    ----------
    sm : sm.SE3
        spatial math transformation
    Returns
    -------
    pose: np.ndarray
        [x, y, z, qw, qx, qy, qz]
    """
    xyz = sm.t
    quat = smb.r2q(sm.R)
    return [*xyz, *quat]


def sg_to_curobo_dict(shapes: List[sg.CollisionShape], env_name):
    """
    Transform a list of spatial geometry shapes to curobo shapes

    Parameters
    ----------
    shapes : List[sg.CollisionShape]
        list of spatial geometry shapes
    """
    cuboids = []
    cylinders = []
    index = 0
    for shape in shapes:

        pose = sm_to_curobo(sm.SE3(shape.T))
        # print(shape.scale)
        if isinstance(shape, sg.Cuboid):
            cuboids.append(Cuboid(name=f"{env_name}_obs_{index}", pose = pose, dims = list(shape.scale)))
            # cuboids.append(cuboid)
        elif isinstance(shape, sg.Cylinder):
            cylinder = Cylinder(name=f"{env_name}_obs_{index}", pose = pose, radius = float(shape.radius), height = float(shape.length))
            cylinders.append(cylinder)
        else:
            raise NotImplementedError(f"Shape {shape} not supported")
        index += 1
    dic = defaultdict(dict)
    for cuboid in cuboids:
        dic["cuboid"][cuboid.name] = {}
        dic["cuboid"][cuboid.name]['pose'] = cuboid.pose.tolist()
        dic["cuboid"][cuboid.name]['dims'] = np.array(cuboid.dims).tolist()
    for cylinder in cylinders:
        dic["cylinder"][cylinder.name] = {}
        dic["cylinder"][cylinder.name]['pose'] = cylinder.pose.tolist()
        dic["cylinder"][cylinder.name]['radius'] = cylinder.radius
        dic["cylinder"][cylinder.name]['height'] = cylinder.height

    print(dic)
    with open("data/curobo_envs/{}.yaml".format(env_name), 'w') as file:
        yaml.dump(dic, file, default_flow_style=False)
    return
    

def sg_to_curobo(shapes: List[sg.CollisionShape]):
    """
    Transform a list of spatial geometry shapes to curobo shapes

    Parameters
    ----------
    shapes : List[sg.CollisionShape]
        list of spatial geometry shapes
    """
    cuboids = []
    cylinders = []
    index = 0
    for shape in shapes:

        pose = sm_to_curobo(sm.SE3(shape.T))
        # print(shape.scale)
        if isinstance(shape, sg.Cuboid):
            cuboids.append(Cuboid(name=f"obs_{index}", pose = pose, dims = list(shape.scale)))
            # cuboids.append(cuboid)
        elif isinstance(shape, sg.Cylinder):
            cylinder = Cylinder(name=f"obs_{index}", pose = pose, radius = shape.radius, height = shape.length)
            cylinders.append(cylinder)
        else:
            raise NotImplementedError(f"Shape {shape} not supported")
        index += 1

    return WorldConfig(cuboid = cuboids, cylinder = cylinders).get_obb_world()





