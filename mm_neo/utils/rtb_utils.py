import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from typing import List


def get_distances(robot: rtb.robot, shapes: List[sg.CollisionShape]) -> dict:
    """
    Get the dict of the distances of each of the links
    Parameters
    ----------
    robot: rtb.robot
        robot
    shapes: List[sg.Shape]
        list of collisions shapes

    Returns
    -------
    result: dict
        {"Link_name": distance}
    """

    result = {}
    links, _, _ = robot.get_path()
    for link in links:
        if len(link.collision) > 0:
            d, grad, po, pa = get_closest_point_from_link(link, shapes)
            result[link.name] = d
    return result


def get_average_distance(robot, shapes):
    """
    Get the average distance of the robot to the shapes
    Parameters
    ----------
    robot: rtb.robot
        robot
    shapes: List[sg.Shapes]
        list of collisions shapes

    Returns
    -------
    result: float
        average distances
    """
    distances = get_distances(robot, shapes)
    return np.mean(list(distances.values()))


def get_closest_point_from_link(link: rtb.Link, shapes, search_distance=10.0):
    min_d = search_distance
    min_po = np.array([0, 0, 0])
    min_pa = np.array([0, 0, 0])
    if len(link.collision) == 0:
        return min_d, np.array([0, 0, 0]), min_po, min_pa
    for col_link in link.collision:
        for i in range(len(shapes)):
            d, po, pa = col_link.closest_point(shapes[i], inf_dist=min_d)
            if d is None:
                continue
            if d < min_d:
                min_d = d
                min_po = po
                min_pa = pa

    if min_d == search_distance:
        print(search_distance)
        print("error")
    return min_d, (min_po - min_pa) / np.linalg.norm(min_po - min_pa), min_po, min_pa


def get_closest_point_from_robot(robot, shapes, search_distance=1000.0):
    """
    Compute the closes distance from the robot to an array of shapes
    :param robot: robot to check the distance
    :param shapes: array of shapes to check the distance
    :param search_distance: maximum distance to check using pybullet
    :return: minimum distance
    """
    min_d = search_distance
    min_po = np.array([0, 0, 0])
    min_pa = np.array([0, 0, 0])
    for link in robot.links:
        d, grad, po, pa = get_closest_point_from_link(link, shapes, search_distance)
        if d < min_d:
            min_d = d
            min_po = po
            min_pa = pa
    return min_d
