from isdf.modules.fc_map import SDFMap, gradient
from isdf.modules.embedding import PostionalEncoding
import torch
import matplotlib.pyplot as plt
import numpy as np
import spatialmath as sm
from isdf.geometry.transform import transform_3D_grid
import time
from functools import wraps
from typing import Tuple
import json
from scipy.interpolate import interpn
import os
import multiprocessing as mp
import skimage  # For marching cubes
import trimesh
from mm_neo.utils import swift_utils as swift_utils
from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset

from typing import Optional
import spatialgeometry as sg

# import dummy_sdf_nn.siren as network

import mm_neo

import pdb


class SDF(torch.nn.Module):
    def __init__(self, folder, transformation, device="cuda", siren=False):
        super().__init__()
        self.device = device
        self.siren = siren
        self.sdf_function = self.load(folder)
        self.transformation = self.get_transformation(transformation)

    def get_transformation(self, transformation: sm.SE3):
        try:
            output = torch.tensor(
                transformation.A,
                device=self.device,
                dtype=torch.float16,
                requires_grad=True,
            )
        except:
            output = torch.tensor(
                transformation,
                device=self.device,
                dtype=torch.float16,
                requires_grad=True,
            )
        return output

    def load_checkpoint(self, file):
        self.sdf_function.load_state_dict(
            torch.load(file, map_location="cpu")["model_state_dict"]
        )
        self.sdf_function.to(self.device).type(torch.float16)
        return

    def load(self, folder: str):
        self.folder_name = folder
        print(folder)
        with open(f"{folder}/config.json", "r") as f:
            params = json.load(f)

        model_params = params["model"]
        encoding_params = model_params["embedding"]
        # model_params["scale_output"] = 1 #/model_params["scale_output"]
        if not self.siren:
            encoding = PostionalEncoding(
                encoding_params["gauss_embed"],
                max_deg=encoding_params["n_embed_funcs"],
                scale=encoding_params["scale_input"],
            ).type(torch.float16)
            encoding = encoding.half()

            self.scale = encoding_params["scale_input"]
            iSDF = SDFMap(
                encoding,
                hidden_size=model_params["hidden_feature_size"],
                hidden_layers_block=model_params["hidden_layers_block"],
                scale_output=model_params["scale_output"],
            )
            filename = f"{folder}/checkpoints/last_step.pth"
        else:
            self.scale = encoding_params["scale_input"]
            iSDF = SDFMap(
                None,
                hidden_size=model_params["hidden_feature_size"],
                hidden_layers_block=model_params["hidden_layers_block"],
                scale_output=model_params["scale_output"],
                siren=True,
            )
            print(model_params["scale_output"])
            filename = f"{folder}/checkpoints_siren/last_step.pth"

        iSDF.load_state_dict(torch.load(filename)["model_state_dict"], strict=False)
        # pdb.set_trace()
        # iSDF = iSDF.half()
        if self.device == "cuda":
            iSDF.cuda()

        return iSDF

    def get_mesh(self, bounds, levels=[0.0]):
        """
        Perform marching cubes on the sdf to get a level set mesh
        """
        # Get 3D grid
        dim = 100
        grid = self.generate_3d_grid(bounds[:, 0], bounds[:, 1], bounds[:, 2], dim)

        grid = grid.to(self.device)
        # Get SDF values
        sdf = self.sdf_forward(grid).detach().cpu()
        sdf = sdf.view(dim, dim, dim).numpy()
        grid.detach().cpu()
        meshes = []

        for level in levels:
            vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(
                sdf,
                level=level,
            )

            dim = sdf.shape[0]
            vertices = vertices / (dim - 1)
            mesh = trimesh.Trimesh(
                vertices=vertices, vertex_normals=vertex_normals, faces=faces
            )

            meshes.append(mesh)

        del grid
        return meshes

    def sdf_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Foward the sdf

        Args:
            tensor (torch.Tensor): Location of the query [n_points,3]

        Returns:
            torch.Tensor: Distance to the closest obstacle [n_points,1]
        """
        input = transform_3D_grid(tensor, transform=self.transformation, scale=None)
        distance = self.sdf_function(input)
        distance = distance.unsqueeze(-1)
        return distance

    def confidence_forward(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance and confidence of the SDF model, only available if
        the model was trained with confidence output

        Args:
            tensor (torch.Tensor): input [n_points,3]

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: distance and confidence [n_points,1]
            for both
        """
        input = transform_3D_grid(
            tensor, transform=self.transformation, scale=self.scale
        )
        distance, confidence = self.sdf_function(input)
        confidence = confidence.squeeze()
        return distance, confidence

    def get_closest_voxel(self, point):
        """ """
        grad, distance_1 = self.get_closest_2(point)
        index = (
            point - self.aabb_i
        ) / self.voxel_size + 0.0001  # ps: I am not proud of this addition
        index = torch.round(index)
        index = index.to(torch.long)

        positons_in_grid = self.aabb_i + index * self.voxel_size

        delta_distance = torch.sum(point - positons_in_grid, dim=-1)

        # distance = self.voxel_grid[]

        distance = self.voxel_grid[index[:, 0], index[:, 1], index[:, 2]]
        distance = distance.reshape(-1)

        distance = distance + delta_distance.cpu()

        distance = distance.detach().cpu()

        interpolated_distance = interpn(
            self.voxel_points_test,
            self.voxel_grid.numpy(),
            point.detach().cpu(),
            method="linear",
        )

        # return grad, result
        return grad, torch.from_numpy(interpolated_distance).reshape(-1)

    def get_closest_2(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the distance of the closest point and the direction
        :param point: torch.tensor points to do the query
        :return gradient, distance:
        """
        augment = point
        output = self.sdf_forward(augment)
        grad = gradient(augment, output).cpu().detach().numpy()
        grad = grad / np.expand_dims(np.linalg.norm(grad, axis=-1), -1)
        return grad[:, :], output[:, :, 0].cpu().detach().reshape(-1)

    def get_closest_w_error(
        self, point: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the orientation to the closest point, its distance
        in addition to an "error" computed as the distance
        after moving along the gradient the predicted distance
        This "error" can be intererpreteed as a map consistency
        :param point: torch.tensor points to do the query
        :return gradient, distance, margin:
        """
        output = self.sdf_forward(point)
        grad = gradient(point, output)
        grad = grad / torch.norm(grad, dim=-1, keepdim=True)

        point_projected = point - output * grad
        output_projected = torch.abs(
            self.sdf_forward(point_projected).detach().cpu().numpy()
        )

        grad = grad.cpu().detach().numpy()

        output -= output_projected

        return (
            grad[:, :],
            output[0, :, :].cpu().detach().reshape(-1),
            output_projected[0, :, :].reshape(-1),
        )

    def get_closest(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the distance of the closest point and the direction
        :param point: torch.tensor points to do the query
        :return gradient, distance:
        """
        output = self.sdf_forward(point)
        grad = gradient(point, output)

        output = output.view(-1)

        return grad, output

    def get_closest_w_error_cuda(
        self, point: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the distance of the closest point and the direction
        :param point: torch.tensor points to do the query
        :return gradient, distance:
        """
        output = self.sdf_forward(point)
        grad = gradient(point, output)
        grad = grad / torch.norm(grad, dim=-1, keepdim=True)
        output = output.view(-1)

        return grad, output

    def get_error(self, point, gradient, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute error of the prediction, this error is computed as the
        inconsistency between the distance and the gradient te equation is:
        f(p')  with p' = p - f(p) * grad(f(p))
        :param point: torch.tensor points to do the query
        :param gradient: torch.tensor gradient of the closest point
        :param distance: torch.tensor distance of the closest point
        :param th: float threshold
        :return error:
        """

        point_projected = point - distance.unsqueeze(-1) * gradient
        output_projected = torch.zeros_like(distance)
        output_projected = (
            self.sdf_forward(point_projected).detach()[:, 0, 0].reshape(-1)
        )

        return output_projected

    def get_error_eikonal(
        self, point, gradient, distance: torch.Tensor, epsilon=1e-5
    ) -> torch.Tensor:
        """
        Compute eikonal loss, this error is computed as performing a small step
        along the gradient direction, and then computing the distance to the
        surface. The equation is:
        e = f(p') - (1 * epsilon)*distance with p' = p - grad(f(p)) * epsilon
        :param point: torch.tensor points to do the query
        :param gradient: torch.tensor gradient of the closest point
        :param distance: torch.tensor distance of the closest point
        :param th: float threshold
        :return error:
        """

        point_projected = point - epsilon * gradient
        output_projected = torch.zeros_like(distance)
        # output_projected[distance>th] = self.sdf_forward(point_projected[distance>th]).detach()[:,0,0].reshape(-1)
        output_projected = (
            self.sdf_forward(point_projected).detach()[:, 0, 0].reshape(-1)
        )

        error = torch.zeros_like(distance)
        error = torch.abs(output_projected - (distance - epsilon))

        return error

    def get_error_cosine(self, point, gradient, distance: torch.Tensor) -> torch.Tensor:
        """
        Get the distance of the closest point and the direction
        and compute
        :param point: torch.tensor points to do the query
        :param gradient: torch.tensor gradient of the closest point
        :param distance: torch.tensor distance of the closest point
        :param th: float threshold
        :return gradient, distance, margin:
        """
        cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        point_projected = point - distance.unsqueeze(-1) * gradient * 0.5

        output_projected = torch.ones_like(distance)  # * 0.9 * distance#
        # output_projected[distance>th] = self.sdf_forward(point_projected[distance>th]).detach()[:,0,0].reshape(-1)
        output_gradient, output_distance = self.get_closest_cuda(point_projected)
        # output_projected[distance>th] = cosine(gradient[distance>th], output_gradient[distance>th])

        output_projected = cosine(gradient, output_gradient)

        return output_projected

    def get_closest_3(self, point: torch.Tensor, steps: int, lamda=0.01):
        grad_tot = torch.zeros_like(point)
        for i in range(steps):
            output = self.sdf_forward(point)
            if i == 0:
                output_init = output
            grad = gradient(point, output)
            grad_tot += grad
            with torch.no_grad():
                point += lamda * grad
        grad_tot = grad_tot / steps
        return grad_tot[:, :].cpu().detach().numpy(), output_init[
            0, :, :
        ].cpu().detach()  # .reshape(-1)

    def get_closest_direction(self, point, delta=0.0001):
        """
        Compute the normalize vector to the closest obstacle from the sample point
        to the shape inside the shapes
        ::
        """

        vector = np.zeros((3))
        x = [-delta, 0, delta]
        y = [-delta, 0, delta]
        z = [-delta, 0, delta]

        query = torch.tensor(
            [point[0], point[1], point[2]], device=self.device, dtype=torch.float32
        ).reshape(1, 3)
        # Could do better than this
        for ix in x:
            for iy in y:
                for iz in z:
                    with torch.no_grad():
                        min_dir = self.sdf_forward(
                            query
                            + torch.tensor([ix, iy, iz], device=self.device).reshape(
                                1, 3
                            )
                        )
                        # print(f"measureded distance {min_dir * 100} cm")
                    vector[0] += min_dir * ix
                    vector[1] += min_dir * iy
                    vector[2] += min_dir * iz

        min_dir = min_dir.cpu().reshape(1)[0]
        return vector / np.linalg.norm(vector), min_dir  # , min_pa, min_po

    def generate_3d_grid(self, bound_x, bound_y, bound_z, n_points=10):
        """
        Generate a 3D field of the SDF
        """

        x = torch.linspace(
            bound_x[0], bound_x[1], n_points, requires_grad=True, dtype=torch.float32
        )
        y = torch.linspace(
            bound_y[0], bound_y[1], n_points, requires_grad=True, dtype=torch.float32
        )
        z = torch.linspace(
            bound_z[0], bound_z[1], n_points, requires_grad=True, dtype=torch.float32
        )
        X, Y, Z = torch.meshgrid(x, y, z)
        points = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), 1)

        del x, y, z, X, Y, Z
        return points

    def generate_3d_voxel_grid(self, bound_i, bound_f, voxel_size=0.05):
        """
        Generate a 3D grid of precomputed sdf values
        params: voxel_size : float
        """

        x = torch.range(
            bound_i[0], bound_f[0], voxel_size, requires_grad=True, dtype=torch.float32
        )
        y = torch.range(
            bound_i[1], bound_f[1], voxel_size, requires_grad=True, dtype=torch.float32
        )
        z = torch.range(
            bound_i[2], bound_f[2], voxel_size, requires_grad=True, dtype=torch.float32
        )

        self.voxel_points_test = (x.detach(), y.detach(), z.detach())

        X, Y, Z = torch.meshgrid(x, y, z)
        points = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), 1)

        distance = self.sdf_forward(points.cuda())

        self.voxel_points = (
            points.reshape((x.shape[0], y.shape[0], z.shape[0], 3)).detach().cpu()
        )
        self.voxel_points.requires_grad = True
        self.voxel_grid = (
            distance.reshape((x.shape[0], y.shape[0], z.shape[0], 1)).detach().cpu()
        )
        self.voxel_size = voxel_size

    def generate_grid(
        self, bound_1, bound_2, fixed_input=1.0, axis=0, n_1=500, n_2=500
    ):
        """
        :param bound_1: list of two floats with [min , max]
        :param bound_2: list of two floats with [min , max]
        :param fixed_input: float with the value for the fixed axis
        :axis: int with the axis to be fixed (0,1,2) -> (x,y,z)
        :param n_1: int with the number of points in the first axis
        :param n_2: int with the number of points in the second axis

        :return xyz: tensor with the points in the grid
        :return axis_list: meshgrid
        """
        assert axis in [0, 1, 2], "Axis must be 0,1 or 2 for x,y,z"

        axis_1 = torch.linspace(bound_1[0], bound_1[1], n_1, requires_grad=True)
        axis_2 = torch.linspace(bound_2[0], bound_2[1], n_2, requires_grad=True)
        axis_3 = torch.ones((n_1, n_2), requires_grad=True) * fixed_input

        axis_list = torch.meshgrid(axis_1, axis_2)

        if axis == 0:
            xyz = torch.cat(
                (
                    axis_3.reshape(-1, 1),
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                ),
                1,
            )
        elif axis == 1:
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                ),
                1,
            )
        else:
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                ),
                1,
            )

        for i in axis_list:
            i.detach().cpu()
        return xyz.cuda(), axis_list

    def plot_error_step(
        self, xyz, grad, distance, axis_list, plot=True, exp_name="test"
    ):
        """
        Compute the error of the sdf at each point of the grid

        Use the step idea to compute the error

        e = |sdf(x')| where x' is x + sdf(x) * d_x of sdf(x)

        """
        error = self.get_error(xyz, grad, distance, 0.01)
        error = (
            error.reshape(axis_list[0].shape, axis_list[1].shape).detach().cpu().numpy()
        )
        distance = (
            distance.reshape(axis_list[0].shape, axis_list[1].shape)
            .detach()
            .cpu()
            .numpy()
        )
        if plot:
            os.makedirs(f"vis/{exp_name}", exist_ok=True)
            fig, ax = plt.subplots()
            a = ax.contour(
                axis_list[0].detach().cpu().numpy(),
                axis_list[1].detach().cpu().numpy(),
                distance,
                levels=0,
            )
            a = plt.scatter(
                axis_list[0].detach().numpy(),
                axis_list[1].detach().numpy(),
                c=error,
                cmap="jet",
                s=1,
            )
            plt.colorbar(a)
            ax.set_title(
                r"Error $e = f(p')$ where $p' = p - \triangledown_x f(p) \cdot f(p)$"
            )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            mp.Process(plt.savefig(f"vis/{exp_name}/error_step.png", dpi=300)).start()
            plt.close(fig)

        return error

    def plot_error_eikonal(
        self, xyz, grad, distance, axis_list, plot=True, exp_name="test"
    ):
        """
        Compute the error of the sdf at each point of the grid

        Use the step idea to compute the error

        e = |sdf(x') - (sdf(x) - epsilon)| where x' is x + e * d_x of sdf(x)

        """
        error = self.get_error_eikonal(xyz, grad, distance, 0.01)
        error = (
            error.reshape(axis_list[0].shape, axis_list[1].shape).detach().cpu().numpy()
        )
        distance = (
            distance.reshape(axis_list[0].shape, axis_list[1].shape)
            .detach()
            .cpu()
            .numpy()
        )
        if plot:
            os.makedirs(f"vis/{exp_name}", exist_ok=True)
            fig, ax = plt.subplots()
            a = ax.contour(
                axis_list[0].detach().cpu().numpy(),
                axis_list[1].detach().cpu().numpy(),
                distance,
                levels=0,
            )
            a = plt.scatter(
                axis_list[0].detach().numpy(),
                axis_list[1].detach().numpy(),
                c=error,
                cmap="jet",
                s=1,
            )
            plt.colorbar(a)
            ax.set_title(
                r"Error $e = \|f(p') - (f(p) - \epsilon)\|$ where $p' = p -\triangledown_xf(p) \cdot \epsilon$"
            )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            mp.Process(
                plt.savefig(f"vis/{exp_name}/error_step_eikonal.png", dpi=300)
            ).start()
            plt.close(fig)

        return error

    def plot_contour(self, xyz, axis_list, distance, exp_name="test", iteration=""):
        """
        Plot the contour of the distance field
        """
        distance = (
            distance.reshape(axis_list[0].shape, axis_list[1].shape)
            .detach()
            .cpu()
            .numpy()
        )
        os.makedirs(f"vis/{exp_name}", exist_ok=True)
        fig, ax = plt.subplots()
        a = ax.contour(
            axis_list[0].detach().cpu().numpy(),
            axis_list[1].detach().cpu().numpy(),
            distance,
        )
        ax.clabel(a, inline=1, fontsize=10)

        ax.set_title("Distance field")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(f"vis/{exp_name}/contour{iteration}.png", dpi=300)
        plt.close(fig)

    def generate_field(
        self,
        bound_1,
        bound_2,
        fixed_input=1.0,
        axis="y",
        n_1=500,
        n_2=500,
        step="",
        exp_name="test",
    ):
        axis_1 = torch.linspace(bound_1[0], bound_1[1], n_1, requires_grad=True)
        axis_2 = torch.linspace(bound_2[0], bound_2[1], n_2, requires_grad=True)
        axis_3 = torch.ones((n_1, n_2), requires_grad=True) * fixed_input

        axis_list = torch.meshgrid(axis_1, axis_2)
        fig, ax = plt.subplots()
        if axis == "x":
            xyz = torch.cat(
                (
                    axis_3.reshape(-1, 1),
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("y [m]")
            plt.ylabel("z [m]")
            plt.title(f"SDF Slice at x = {fixed_input:.2f}_{step}")
            grad, distances = self.get_closest_cuda(xyz.cuda())

            a = ax.contour(
                axis_list[0].detach().cpu().numpy(),
                axis_list[1].detach().cpu().numpy(),
                distances.detach().cpu().numpy(),
                levels=np.array(
                    [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ),
            )
            ax.clabel(a, inline=1, fontsize=10)
            os.makedirs(f"vis/{exp_name}", exist_ok=True)
            mp.Process(
                plt.savefig(f"vis/{exp_name}/{axis}_sdf_slice_{step}.png", dpi=300)
            ).start()
            plt.close("all")

        elif axis == "y":
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                    axis_list[0].reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.title(f"self.sdf_Slice at y = {fixed_input}")
            with torch.no_grad():
                distances = self.sdf_forward(xyz.cuda()).view(n_1, n_2)
            a = plt.scatter(
                axis_list[0], axis_list[1], c=distances.cpu().numpy(), cmap="jet"
            )
            plt.colorbar(a)

        elif axis == "z":
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("y [m]")
            plt.ylabel("x [m]")
            plt.title(f"SDF Slice at z = {fixed_input}")
            grad, distances = self.get_closest_2(xyz.cuda())
            distances = distances.reshape(n_1, n_2)
            grad = grad.reshape(n_1, n_2, 3)
            a = ax.contour(
                axis_list[0].detach().numpy(),
                axis_list[1].detach().numpy(),
                distances,
                levels=np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4]),
            )
            ax.clabel(a, inline=1, fontsize=10)

        # os.makedirs(f"vis/{exp_name}", exist_ok=True)
        # mp.Process(plt.savefig(f"vis/{exp_name}/{axis}_sdf_slice_{step}.png", dpi=300)).start()
        plt.close("all")

    def generate_image(
        self, bound_1, bound_2, fixed_input=1.0, axis="y", n_1=500, n_2=500
    ):
        axis_1 = torch.linspace(bound_1[0], bound_1[1], n_1)
        axis_2 = torch.linspace(bound_2[0], bound_2[1], n_2)
        axis_3 = torch.ones((n_1, n_2)) * fixed_input

        axis_list = torch.meshgrid(axis_1, axis_2)

        if axis == "x":
            xyz = torch.cat(
                (
                    axis_3.reshape(-1, 1),
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("y [m]")
            plt.ylabel("z [m]")
            plt.title(f"SDF Slice at x = {fixed_input}")
            with torch.no_grad():
                distances = self.sdf_forward(xyz.cuda())
            a = plt.scatter(
                axis_list[0], axis_list[1], c=distances.cpu().numpy(), cmap="jet"
            )
            plt.colorbar(a)

        elif axis == "y":
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                    axis_list[0].reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.title(f"self.sdf_Slice at y = {fixed_input}")
            with torch.no_grad():
                distances = self.sdf_forward(xyz.cuda()).view(n_1, n_2)
            a = plt.scatter(
                axis_list[0], axis_list[1], c=distances.cpu().numpy(), cmap="jet"
            )
            plt.colorbar(a)

        elif axis == "z":
            xyz = torch.cat(
                (
                    axis_list[0].reshape(-1, 1),
                    axis_list[1].reshape(-1, 1),
                    axis_3.reshape(-1, 1),
                ),
                1,
            )
            plt.xlabel("y [m]")
            plt.ylabel("x [m]")
            plt.title(f"SDF Slice at z = {fixed_input}")
            plt.xlim(bound_1[1], bound_1[0])
            with torch.no_grad():
                distances = self.sdf_forward(xyz.cuda()).view(n_1, n_2)
            a = plt.scatter(
                axis_list[1], axis_list[0], c=distances.cpu().numpy(), cmap="jet"
            )
            plt.colorbar(a)

        plt.savefig(f"vis/{axis}_sdf_slice__{fixed_input:.4f}_.png")
        plt.figure()


class dummy_SDF(SDF):
    """
    This is a dummy trained SDF model with heavy supervision
    """

    def __init__(self, model_path, transformation=sm.SE3(), device="cuda"):
        super().__init__(model_path, transformation, device)

    def load(self, model_path):
        """
        Load the model
        """
        self.scale = 1

        sdf_function = network.dummy_sdf(3, 256)
        sdf_function.load_state_dict(torch.load(model_path, map_location=self.device))
        sdf_function.to(self.device)
        return sdf_function


def load_isdf_scannet(problem_name, transform=sm.SE3()) -> SDF:
    """Load isdf model trained on scannet datasets

    Args:
        problem_name (str): env name

    Returns:
        SDF: instance of the model
    """
    isdf_folder = os.path.join(
        os.path.dirname(mm_neo.__file__), "../results/isdf", problem_name, "0"
    )

    net = SDF(isdf_folder, transform)
    return net


def load_isdf(problem_name, extra_transformation=None, siren=False) -> SDF:
    """
    Load iSDF model trained with depth images

    Parameters
    ----------
    problem_name
        Name of the env to be loaded
    dataset_folder
        Folder where the dataset is stored
    extra_transformation
        Transformation to be applied to the model to align with the world SE3
    Returns
    -------
    iSDF
        Instance of the model to be used
    """
    isdf_folder = os.path.join(
        os.path.dirname(mm_neo.__file__), "../results/isdf", problem_name, "0"
    )

    with open(os.path.join(isdf_folder) + "/config.json", "r") as f:
        config = json.load(f)

    transform_json_folder = config["dataset"]["seq_dir"]

    with open(os.path.join(transform_json_folder, "transforms.json"), "r") as f:
        dataset_params = json.load(f)

    aabb_i = np.array(dataset_params["aabb"][0])
    aabb_f = np.array(dataset_params["aabb"][1])

    center = (aabb_i + aabb_f) / -2
    transformation = sm.SE3(center[0], center[1], center[2])

    if extra_transformation is not None:
        transformation = transformation * extra_transformation

    net = SDF(isdf_folder, transformation, siren=siren)
    return net


def load_sdf_swift(
    env_name: str, dataset_type: str, pose: Optional[sm.SE3] = None, siren=False
) -> Tuple[sg.Mesh, SDF]:
    """Load sdf model getting the mesh and the sdf from the swift dataset

    Parameters
    ----------
    env_name : str
        name of the environment to load
    dataset_type : str
        dataset type options = ["ARKit", "Replica"]

    Returns
    -------
    [sg.Mesh, SDF]
        Mesh of the zero level set of the sdf model, SDF instance of the model
    """

    mm_neo_path = os.path.dirname(mm_neo.__file__) + "/../"

    if siren:
        mesh_file_template = mm_neo_path + "/results/isdf/{}/0/meshes_siren/last.stl"
    else:
        mesh_file_template = mm_neo_path + "/results/isdf/{}/0/meshes/last.stl"
    reconstruction = swift_utils.load_mesh(
        mesh_file_template.format(env_name), scale=1.0, color=[1, 0.4, 0, 0.5]
    )

    sdf_folder = mm_neo_path + "/results/isdf/{}/0/config.json"
    with open(sdf_folder.format(env_name)) as f:
        config = json.load(f)

    # ** The inv_bounds_transform is the transform from the center of the scene in order to match the mesh
    # ** The fix_frame is the transform to move the mesh to the correct position, therefore
    # ** Taking the inverse of the fix_frame and multiplying by the inv_bounds_transform we get the correct transform

    if dataset_type == "Replica":
        fix_frame = sm.SE3(0, 0, 0.1) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Ry(np.pi / 2)
        dataset = ReplicaDataset(
            config["dataset"], parent_path=os.environ["SDF"], load_gt=True
        )
        sdf = load_isdf_scannet(
            env_name, dataset.inv_bounds_transform @ np.linalg.inv(fix_frame)
        )
        reconstruction.T = fix_frame
    elif dataset_type == "ARKit":
        fix_frame = sm.SE3(0.73, 0, 1.1) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Ry(-np.pi / 2)
        if pose is not None:
            fix_frame = pose

        sdf = load_isdf(
            env_name, extra_transformation=np.linalg.inv(fix_frame), siren=siren
        )
        reconstruction.T = fix_frame
    elif dataset_type == "Blender":
        folder = config["dataset"]["seq_dir"]
        sdf = load_isdf(env_name, dataset_folder=folder)

    return reconstruction, sdf


def load_dummy_sdf(
    env_name: str,
    folder: str = "results/dummy-sdf",
    pose: Optional[sm.SE3] = sm.SE3(),
    eikonal=True,
) -> Tuple[sg.Mesh, SDF, sg.Mesh]:
    """Load dummy sdf model getting the mesh and the sdf from the swift dataset

    Parameters
    ----------
    env_name : str
        name of the environment to load

    Returns
    -------
    [sg.Mesh, SDF, sg.Mesh]
        Mesh of the zero level set of the sdf model, SDF instance of the model
        and the mesh obtained as GT
    """

    mesh_file_template = folder + "/meshes/{}_pred.stl"
    reconstruction = swift_utils.load_mesh(
        mesh_file_template.format(env_name), scale=1.0, color=[1, 0.0, 0, 0.5]
    )

    gt_mesh_file_template = folder + "/meshes/{}_gt.stl"
    gt_reconstruction = swift_utils.load_mesh(
        gt_mesh_file_template.format(env_name), scale=1.0, color=[0, 0.4, 1, 0.5]
    )

    if eikonal:
        sdf_folder = folder + "/checkpoints/eikonal_{}.pth"
    else:
        sdf_folder = folder + "/checkpoints/{}.pth"
    reconstruction.T = pose
    sdf = dummy_SDF(sdf_folder.format(env_name), transformation=pose.inv())

    return reconstruction, sdf, gt_reconstruction


if __name__ == "__main__":
    pass
