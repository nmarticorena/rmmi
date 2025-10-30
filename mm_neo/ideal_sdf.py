# Idea of performing a SDF check by torch
# The sdf functions comes form https://iquilezles.org/articles/distfunctions/
"""
p is the sampled point, h is the heigth of the cylinder and r is the radius
float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdBox( vec3 p, vec3 b ) b is [x,y,z] directions
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

notes: y is z in gl


"""

try:
    import rospy
    from std_msgs.msg import Float32MultiArray
except:
    Float32MultiArray = None
    print("ros not available")
import json
import spatialgeometry as sg
import spatialmath as sm
import numpy as np
import os

try:
    from mm_neo.sdf import SDF
except:
    SDF = None


from typing import List, Tuple
import torch
import pdb


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def length(vec: torch.Tensor):
    """
    OpenGL length function on torch
    """
    return torch.norm(vec, dim=-1)


def total(input: torch.Tensor, objects: List[torch.nn.Module]) -> torch.Tensor:
    """
    Perform total SDF of a set of objects
    """
    results = [o(input) for o in objects]
    # for o in objects:
    # results.append(o.distance(input))

    return composite(results)


def composite(tensors) -> torch.Tensor:
    distance = torch.vstack(tensors).min(axis=0)[0]
    return distance


class SphereSDF(torch.nn.Module):
    def __init__(self, radius, origin):
        """
        SphereSDF
        Parameters
        ----------
        radius: float
            radius in m of the sphere
        origin: array
            [3] location of center of box in x,y,z
        """
        super(SphereSDF, self).__init__()
        self.radius = torch.tensor(radius, dtype=torch.float32, device="cuda")
        self.origin = torch.tensor(origin, dtype=torch.float32, device="cuda")

    def forward(self, points: torch.Tensor):
        x = points - self.origin
        return length(x) - self.radius


class BoxSDF(torch.nn.Module):
    def __init__(self, scale, position):
        """
        BoxSDF
        Parameters
        ----------
        size: array
            [3] size in x,y,z
        origin: array
            [3] location of center of box in x,y,z
        """
        super(BoxSDF, self).__init__()
        self.size = torch.tensor(scale, dtype=torch.float32, device="cuda")
        self.origin = torch.tensor(position, dtype=torch.float32, device="cuda")

    def forward(self, points: torch.Tensor):
        x = points - self.origin
        q = torch.abs(x) - self.size
        return length(torch.maximum(q, torch.zeros_like(q))) + torch.minimum(
            torch.maximum(q[:, 0], torch.maximum(q[:, 1], q[:, 2])),
            torch.zeros_like(q[:, 0]),
        )


class BCylinderSDF(torch.nn.Module):
    def __init__(self, r, origin):
        super(BCylinderSDF, self).__init__()
        self.r = torch.tensor([[r]], dtype=torch.float32, device="cuda")
        self.origin = torch.tensor(origin, dtype=torch.float32, device="cuda")

    def forward(self, p: torch.Tensor):
        # import pdb
        pxy = torch.stack([p[:, 1], p[:, 2]], dim=-1)
        # pdb.set_trace()
        return -1 * (length(pxy - self.origin[1:3]) - self.r)


class CylinderSDF(torch.nn.Module):
    def __init__(self, heigth, radius, position):
        super(CylinderSDF, self).__init__()
        self.rh = torch.tensor([[radius, heigth]], dtype=torch.float32, device="cuda")
        self.origin = torch.tensor(
            position, dtype=torch.float32, device="cuda"
        )  # , requires_grad= True)

    def forward(self, points: torch.Tensor):
        """
        Compute distance of set of points to the cylinder
        Parameters
        ----------
        points: torch.Tensor
            [N,3] x,y,z location of the diferent queried points
        Returns
        -------
        distance: torch.Tensor
            [N,1] SDF Euclidean distance of the points to the cilinder
        """
        x = points - self.origin
        xz = torch.norm(torch.stack([x[:, 0], x[:, 1]], dim=-1), dim=1)
        d = torch.abs(torch.stack([xz, x[:, 2]], dim=-1)) - self.rh
        m = torch.max(d, dim=1)[0]
        rigth = torch.norm(torch.maximum(d, torch.zeros_like(d)), dim=-1)
        return torch.minimum(m, torch.zeros((x.shape[0]), device="cuda")) + rigth


class RosCylinderSDF:
    def __init__(self, topic: str, sdf: CylinderSDF):
        self.sdf = sdf
        rospy.Subscriber(topic, Float32MultiArray, self.update)
        # msg = rospy.wait_for_message(topic, Float32MultiArray)
        # self.update(msg)

    def update(self, msg: Float32MultiArray):
        self.sdf.rh[0, 0] = msg.data[-2]
        self.sdf.rh[0, 1] = msg.data[-1]
        self.sdf.origin[0] = msg.data[0]
        self.sdf.origin[1] = msg.data[1]
        self.sdf.origin[2] = msg.data[2]


class RosBoxSDF:
    def __init__(self, topic: str, sdf: BoxSDF):
        self.sdf = sdf
        rospy.Subscriber(topic, Float32MultiArray, self.update)

    def update(self, msg: Float32MultiArray):
        self.sdf.origin[0] = msg.data[0]
        self.sdf.origin[1] = msg.data[1]
        self.sdf.origin[2] = msg.data[2]

        self.sdf.size[0] = msg.data[-3]
        self.sdf.size[1] = msg.data[-2]
        self.sdf.size[2] = msg.data[-1]


class IdealSDF:
    def __init__(self, sdfs):
        self.sdf = sdfs

    def get_distance(self, p: torch.Tensor):
        d = total(p, self.sdf)
        return d

    def get_closest(self, p: torch.Tensor):
        """
        Get gradient and distance

        Parameters
        ----------
        p: torch.Tensor
            point in space [N,3]
        Returns
        -------
        grad: torch.Tensor
            gradient to the direction of increasing distance [N,3]
        d: torch.Tensor
            distance to closest object [N,1]

        """
        d = total(p, self.sdf)
        grad = gradient(p, d)
        return grad, d


class CombinedSDF:
    def __init__(self, ideal_sdfs, sdf: SDF):
        self.ideal_sdfs = ideal_sdfs
        self.sdf = sdf

    def get_distance(self, p: torch.Tensor):
        d = total(p, self.ideal_sdfs)
        d2 = self.sdf.sdf_forward(p)
        d = torch.minimum(d, d2)
        return d

    def get_closest(self, p: torch.Tensor):
        """
        Get gradient and distance

        Parameters
        ----------
        p: torch.Tensor
            point in space [N,3]
        Returns
        -------
        grad: torch.Tensor
            gradient to the direction of increasing distance [N,3]
        d: torch.Tensor
            distance to closest object [N,1]

        """
        import pdb

        # pdb.set_trace()
        d = total(p, self.ideal_sdfs)
        d2 = self.sdf.sdf_forward(p).squeeze()
        d = torch.minimum(d, d2)
        grad = gradient(p, d)
        return grad, d


class SwiftBox:
    def __init__(self, env, params, headless=False):
        """
        Parameters
        ----------
        env: swift.Swift
            swift environment
        params: dict
            dictionary with the parameters of the box
            {position: [x,y,z], scale: [x,y,z], s_pos: [x,y,z], s_scale: [x,y,z]}
        headless: bool
            if the simulation is headless default False
        """
        position = params["position"]
        scale = params["scale"]
        # scale = [i *2 for i in scale]
        s_pos = params["s_pos"]
        s_scale = params["s_scale"]
        self.sdf = BoxSDF(scale, position)
        self.sg = sg.Cuboid(scale, pose=sm.SE3(position), color=np.random.rand(3))
        self.mu_pose = np.array(position)
        self.sigma_pose = np.array(s_pos)
        self.mu_scale = np.array(scale)
        self.sigma_scale = np.array(s_scale)
        if not headless:
            env.add(self.sg)

    def sample(self):
        xyz = np.random.uniform(
            self.mu_pose - self.sigma_pose, self.mu_pose + self.sigma_pose
        )
        scale = np.random.uniform(
            self.mu_scale - self.sigma_scale, self.mu_scale + self.sigma_scale
        )

        self.sg.T = sm.SE3(xyz)
        self.sg.scale = [i * 2 for i in scale]

        self.sdf.size = torch.from_numpy(scale).cuda()
        self.sdf.origin = torch.from_numpy(xyz).cuda()
        # print(xyz)
        self.xyz = xyz
        self.scale = scale
        return

    def load(self, info):
        """
        Load the state of the obstacles and the target from a json file
        Parameters
        ----------
        info: dict
            dictionary with the parameters of the box
            {position: [x,y,z], scale: [x,y,z]}
        """
        self.sg.T = sm.SE3(info["position"])
        self.sg.scale = [i * 2 for i in info["scale"]]

        self.sdf.size = torch.tensor(info["scale"], device="cuda")
        self.sdf.origin = torch.tensor(info["position"], device="cuda")
        return

    def get_json(self):
        return {"position": self.xyz.tolist(), "scale": self.scale.tolist()}


class SwiftPoses:
    def __init__(self, env, params, headless=False):
        position = params["position"]
        s_pos = params["s_pos"]
        self.sg = sg.Axes(0.1, pose=sm.SE3(position))
        self.mu_pose = np.array(position)
        self.sigma_pose = np.array(s_pos)
        self.rot = sm.SO3.Rx(np.pi)
        self.pose = sm.SE3.Rt(t=position, R=self.rot)
        if not headless:
            env.add(self.sg)

    def sample(self):
        xyz = np.random.uniform(
            self.mu_pose - self.sigma_pose, self.mu_pose + self.sigma_pose
        )
        self.pose = sm.SE3.Rt(t=xyz, R=self.rot)

        self.sg.T = self.pose
        return


class SwiftCylinder:
    def __init__(self, env, params, headless=False):
        """
        Parameters
        ----------
        env: swift.Swift
            swift environment
        params: dict
            dictionary with the parameters of the box
            {position: [x,y,z], rh: [r,h], pos_mu: [x,y,z], scale_mu: [x,y,z]}
        headless: bool
            if the simulation is headless default False
        """
        position = params["position"]
        s_pos = params["s_pos"]
        radius = params["radius"]
        heigth = params["heigth"]
        s_radius = params["s_scale"][0]
        s_heigth = params["s_scale"][1]
        self.sdf = CylinderSDF(heigth, radius, position)
        self.sg = sg.Cylinder(
            radius, heigth * 2, pose=sm.SE3(position), color=np.random.rand(3)
        )
        self.position = np.array(position)
        self.radius = radius
        self.heigth = heigth
        self.s_pos = np.array(s_pos)
        self.s_radius = s_radius
        self.s_heigth = s_heigth
        if not headless:
            env.add(self.sg)

    def sample(self):
        xyz = np.random.uniform(self.position - self.s_pos, self.position + self.s_pos)
        radius = np.random.uniform(
            self.radius - self.s_radius, self.radius + self.s_radius
        )
        heigth = np.random.uniform(
            self.heigth - self.s_heigth, self.heigth + self.s_heigth
        )

        self.sg.radius = radius
        self.sg.length = heigth

        self.sg.T = sm.SE3(xyz) * sm.SE3.Rx(np.pi / 2)
        self.sdf.rh = torch.tensor([radius, heigth], device="cuda")
        self.sdf.origin = torch.tensor(xyz, device="cuda")

        self.xyz = xyz
        return

    def load(self, info):
        """
        Load the state of the obstacles and the target from a json file
        Parameters
        ----------
        info: dict
            dictionary with the parameters of the box
            {position: [x,y,z], radius: r, heigth: h}
        """
        self.sg.T = sm.SE3(info["position"]) * sm.SE3.Rx(np.pi / 2)
        self.sg.radius = info["radius"]
        self.sg.length = info["heigth"]

        self.sdf.rh = torch.tensor([info["radius"], info["heigth"]], device="cuda")
        self.sdf.origin = torch.tensor(info["position"], device="cuda")
        return

    def get_json(self):
        return {
            "position": self.xyz.tolist(),
            "radius": self.sg.radius,
            "heigth": self.sg.length,
        }


class RandomizedSDF:
    def __init__(self, json_path, env, headless=False):
        """
        Load json with description of the obstacles
        Parameters
        ----------
        json_path: str
            path to the json file
        env: swift.Swift
            swift environment
        headless: bool
            if the simulation is headless default False
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        self.folder = json_path.split("/")[:-1]
        self.folder = "/".join(self.folder)
        swiftSDF = []
        for k, v in data["obstacles"].items():
            print(k)
            if k == "cubes":
                for cube in v:
                    swiftSDF.append(SwiftBox(env, cube, headless))
            if k == "cylinders":
                for cil in v:
                    for _ in range(cil["repeat"]):
                        swiftSDF.append(SwiftCylinder(env, cil, headless))
        self.headless = headless
        target = data["target"]
        self.rotation = np.array(data["target"])[:3, :3]
        self.sdf = IdealSDF([sdf.sdf for sdf in swiftSDF])
        self.swiftSDF = swiftSDF
        target = np.array(data["target"])
        self.target = sm.SE3(target, check=False).norm()
        self.original_target = sm.SE3(target, check=False).norm()
        self.target_axes = sg.Axes(0.1, pose=self.target)
        self.base_xyz = np.array(data["robot_base"]["position"])
        self.base_sigma = np.array(data["robot_base"]["s_position"])
        self.base = sm.SE3(self.base_xyz)
        if not self.headless:
            env.add(self.target_axes)

    def get_files(self):
        sampled = os.listdir(f"{self.folder}")
        sampled = [s for s in sampled if "table" in s]

        sampled = [f"{self.folder}/{s}" for s in sampled]
        return sampled

    def sample(self):
        for sdf in self.swiftSDF:
            sdf.sample()
        # random_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        self.target = self.original_target * sm.SE3.Rz(0)
        self.target_axes.T = self.target
        xyz_base = np.random.uniform(
            self.base_xyz - self.base_sigma, self.base_xyz + self.base_sigma
        )
        # theta_z = np.random.uniform(0,2*np.pi)
        self.base = sm.SE3(xyz_base)
        return

    def save(self, filename):
        """
        Save the current state of the obstacles and the target in a json file
        Parameters
        ----------
        filename: str
            path to the json file
        """
        data = {
            "cubes": [
                sdf.get_json() for sdf in self.swiftSDF if isinstance(sdf, SwiftBox)
            ],
            "cylinders": [
                sdf.get_json()
                for sdf in self.swiftSDF
                if isinstance(sdf, SwiftCylinder)
            ],
            "target": self.target.A.tolist(),
            "robot_base": self.base.A.tolist(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, filename):
        """
        Load the state of the obstacles and the target from a json file
        Parameters
        ----------
        filename: str
            path to the json file
        """
        with open(filename, "r") as f:
            data = json.load(f)
        target = np.array(data["target"])
        self.target = sm.SE3(target, check=False).norm()
        self.target_axes.T = self.target
        base = np.array(data["robot_base"])
        self.base = sm.SE3(base)
        cylinders = data["cylinders"]
        cubes = data["cubes"]
        cyl_index = 0
        cube_index = 0
        for sdf in self.swiftSDF:
            if isinstance(sdf, SwiftCylinder):
                sdf.load(cylinders[cyl_index])
                cyl_index += 1
            if isinstance(sdf, SwiftBox):
                sdf.load(cubes[cube_index])
                cube_index += 1
        return

    def get_obstacles(self):
        return [sdf.sg for sdf in self.swiftSDF]


class RandomizedSDFMultiple:
    def __init__(self, json_path, env, headless=False):
        """
        Load json with description of the obstacles
        Parameters
        ----------
        json_path: str
            path to the json file
        env: swift.Swift
            swift environment
        headless: bool
            if the simulation is headless default False
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        self.folder = json_path.split("/")[:-1]
        self.folder = "/".join(self.folder)
        swiftSDF = []
        for k, v in data["obstacles"].items():
            print(k)
            if k == "cubes":
                for cube in v:
                    swiftSDF.append(SwiftBox(env, cube, headless))
            if k == "cylinders":
                for cil in v:
                    for _ in range(cil["repeat"]):
                        swiftSDF.append(SwiftCylinder(env, cil, headless))
        self.headless = headless
        self.rotation = np.array(data["target"])[:3, :3]
        self.sdf = IdealSDF([sdf.sdf for sdf in swiftSDF])
        self.swiftSDF = swiftSDF
        self.targets_gen = [SwiftPoses(env, t, headless) for t in data["targets"]]

        self.targets = self.get_targets()
        self.base_xyz = np.array(data["robot_base"]["position"])
        self.base_sigma = np.array(data["robot_base"]["s_position"])
        self.base = sm.SE3(self.base_xyz)

    def get_targets(self):
        poses = [t.pose for t in self.targets_gen]
        return poses

    def get_files(self):
        sampled = os.listdir(f"{self.folder}")
        sampled = [s for s in sampled if "table" in s]

        sampled = [f"{self.folder}/{s}" for s in sampled]
        return sampled

    def sample(self):
        for sdf in self.swiftSDF:
            sdf.sample()
        for poses in self.targets_gen:
            poses.sample()
        self.targets = self.get_targets()
        xyz_base = np.random.uniform(
            self.base_xyz - self.base_sigma, self.base_xyz + self.base_sigma
        )
        self.base = sm.SE3(xyz_base)
        return

    def save(self, filename):
        """
        Save the current state of the obstacles and the target in a json file
        Parameters
        ----------
        filename: str
            path to the json file
        """
        data = {
            "cubes": [
                sdf.get_json() for sdf in self.swiftSDF if isinstance(sdf, SwiftBox)
            ],
            "cylinders": [
                sdf.get_json()
                for sdf in self.swiftSDF
                if isinstance(sdf, SwiftCylinder)
            ],
            "targets": [target.pose.A.tolist() for target in self.targets_gen],
            "robot_base": self.base.A.tolist(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, filename):
        """
        Load the state of the obstacles and the target from a json file
        Parameters
        ----------
        filename: str
            path to the json file
        """
        with open(filename, "r") as f:
            data = json.load(f)
        self.targets = [
            sm.SE3(np.array(target), check=False).norm() for target in data["targets"]
        ]
        for i in range(len(self.targets)):
            self.targets_gen[i].pose = self.targets[i]
            self.targets_gen[i].sg.T = self.targets[i]
        base = np.array(data["robot_base"])
        self.base = sm.SE3(base)
        cylinders = data["cylinders"]
        cubes = data["cubes"]
        cyl_index = 0
        cube_index = 0
        for sdf in self.swiftSDF:
            if isinstance(sdf, SwiftCylinder):
                sdf.load(cylinders[cyl_index])
                cyl_index += 1
            if isinstance(sdf, SwiftBox):
                sdf.load(cubes[cube_index])
                cube_index += 1

        return

    def get_obstacles(self):
        return [sdf.sg for sdf in self.swiftSDF]


def load_from_json(json_path) -> Tuple[IdealSDF, List[sg.CollisionShape], sm.SE3]:
    """
    Load an ideal sdf from a json path

    Parameters
    ----------
    json_path : str
        string as a path to the json file containing the dataset info
    Returns
    -------
    Tuple[IdealSDF, List[sg.CollisionShape], sm.SE3]
        [sdf, list of the shapes, target pose]
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    try:
        obstacles = data["obstacles"]
    except KeyError:
        obstacles = data
    sdf = []
    sgs = []
    for k, v in obstacles.items():
        print(k)
        if k == "cubes":
            for cube in v:
                sdf.append(BoxSDF(**cube))
                scale = cube["scale"]
                scale = [i * 2 for i in scale]
                pose = sm.SE3(cube["position"]).norm()
                sgs.append(sg.Cuboid(scale, pose=pose, color=np.random.rand(3)))
        if k == "cylinders":
            for cil in v:
                sdf.append(CylinderSDF(**cil))
                pose = sm.SE3(cil["position"]).norm()
                sgs.append(
                    sg.Cylinder(
                        cil["radius"],
                        cil["heigth"] * 2,
                        pose=pose,
                        color=np.random.rand(3),
                    )
                )

    sdf_model = IdealSDF(sdf)
    if "target" in data:
        objective = sm.SE3(np.array(data["target"]), check=False).norm()
    elif "targets" in data:
        objective = [sm.SE3(np.array(t), check=False).norm() for t in data["targets"]]
    else:
        objective = sm.SE3()
    return sdf_model, sgs, objective


def load_from_json_full(
    json_path,
) -> Tuple[IdealSDF, List[sg.CollisionShape], sm.SE3, sm.SE3]:
    """
    Load an ideal sdf from a json path

    Parameters
    ----------
    json_path : str
        string as a path to the json file containing the dataset info
    Returns
    -------
    Tuple[IdealSDF, List[sg.CollisionShape], sm.SE3]
        [sdf, list of the shapes, target pose]
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    try:
        obstacles = data["obstacles"]
    except KeyError:
        obstacles = data
    sdf = []
    sgs = []
    for k, v in obstacles.items():
        print(k)
        if k == "cubes":
            for cube in v:
                sdf.append(BoxSDF(**cube))
                scale = cube["scale"]
                scale = [i * 2 for i in scale]
                pose = sm.SE3(cube["position"]).norm()
                sgs.append(sg.Cuboid(scale, pose=pose, color=np.random.rand(3)))
        if k == "cylinders":
            for cil in v:
                sdf.append(CylinderSDF(**cil))
                pose = sm.SE3(cil["position"]).norm()
                sgs.append(
                    sg.Cylinder(
                        cil["radius"],
                        cil["heigth"] * 2,
                        pose=pose,
                        color=np.random.rand(3),
                    )
                )

    sdf_model = IdealSDF(sdf)
    if "target" in data:
        objective = sm.SE3(np.array(data["target"]), check=False).norm()
    elif "targets" in data:
        objective = [sm.SE3(np.array(t), check=False).norm() for t in data["targets"]]
    else:
        objective = sm.SE3()

    base = sm.SE3(np.array(data["robot_base"]), check=False)
    base = base.norm()
    return sdf_model, sgs, objective, base


