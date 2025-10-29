# TODO Add config file for controller parameters
import time
import math

from typing import Optional, List, Tuple

import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift

import torch
import numpy as np
import qpsolvers as qp

from neural_robot.robot import NeuralRobot
import neural_robot.utils.math as helper_math

from mm_neo.configs.controllers import NeoConfig, Slack, Manipulability
from mm_neo.logger import Logger
from mm_neo.utils import rtb_utils
from mm_neo.sdf import SDF
# Ros
try:
    from geometry_msgs.msg import Pose
    from mm_neo.debug_tools import PointsRvizVisualizer

except ImportError:
    Pose = None
    pass

np.set_printoptions(precision=3, suppress=True)


def compute_collision_constraints(
        robot:NeuralRobot, g_w:torch.Tensor, distance:torch.Tensor, controller_config:NeoConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the collision constraints for the QP problem.
    Args:
        g_w (torch.Tensor): The spatial gradient of the distance in the world frame [n,3].
        distances (torch.Tensor): Each of the distances [n,1].
    Returns:
        torch.Tensor: The Ain matrix for the QP problem. [n*+6, n*+6]
        torch.Tensor: The bin matrix for the QP problem. [n*+6]
        where n* is the len(distances<config.di)
    """

    Jac = robot.get_jacobians_collision()

    mask = distance < controller_config.di

    g_w = g_w[mask].clone()
    d = distance[mask]
    Jac = Jac[mask]
    g_b = helper_math.rotate_vec(g_w, robot.transform)
    norm_h = torch.zeros(g_b.shape[0], 6).cuda()
    norm_h[:, :3] = g_b
    c_Ain = -1 * norm_h.unsqueeze(1) @ Jac
    c_bin = (
        controller_config.xi
        * (d - controller_config.ds)
        / (controller_config.di - controller_config.ds)
    )
    c_Ain = c_Ain.squeeze()
    if len(c_Ain.shape) == 1:
        c_Ain = c_Ain.unsqueeze(0)


    return c_Ain, c_bin


class MMNeo:
    def __init__(
        self,
        robot: rtb.Robot,
        as_one: bool = False,
        config: NeoConfig = NeoConfig(),
        logger: Optional[Logger] = None,
    ) -> None:
        self.robot = robot
        self.n = robot.n
        self.config = config
        self.logger = logger
        self.as_one = as_one
        self.last_qd = np.zeros(self.n)

    def step(self, Tep: sm.SE3, collisions: Optional[List[sg.CollisionShape]] = None):
        T_We = self.robot.fkine(self.robot._q)

        T_eEp = np.linalg.inv(Tep) * T_We

        # spatial error
        et = np.sum(np.abs(T_eEp[0:3, 3]))   + 1e-10
        # print("error:", et)

        # Gain term (labda) for control minimisation
        Y = self.config.lamda_q

        # Quadratic component of objective function
        Q = np.eye(self.robot.n + 6)

        # Joint velocity component of Q
        Q[: self.robot.n, : self.robot.n] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[self.n :, self.n :] = (1.0 / et) * np.eye(6)

        v, _ = rtb.p_servo(T_We, Tep, self.config.beta, method="rpy")

        v[3:] *= 1.3
        # v = v / 20.0
        # The equality contraints
        Aeq = np.c_[self.robot.jacobe(self.robot._q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = self.config.ps

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = self.config.pi

        # Form the joint limit velocity damper
        Ain[: self.n, : self.n], bin[: self.n] = self.robot.joint_velocity_damper(
            ps, pi, self.n
        )

        if collisions is not None:
            if self.as_one:
                c_Ain, c_bin = self.robot.link_collision_damper_merge(
                    collisions,
                    self.robot.q,
                    0.3,
                    0.05,
                    1,
                    end=self.robot.link_dict["frankie_hand"],
                )
                if c_Ain is not None and c_bin is not None:
                    c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

                    # Stack the inequality constraints
                    Ain = np.r_[Ain, c_Ain]
                    bin = np.r_[bin, c_bin]
            else:
                for collision in collisions:
                    # Form the velocity damper inequality contraint for each collision
                    # object on the robot to the collision in the scene
                    c_Ain, c_bin = self.robot.link_collision_damper(
                        collision,
                        self.robot.q,
                        0.3,
                        0.05,
                        1,
                        end=self.robot.link_dict["frankie_hand"],
                    )  # [collision_links, dof+ 6]

                    # If there are any parts of the robot within the influence distance
                    # to the collision in the scene
                    if c_Ain is not None and c_bin is not None:
                        c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

                        # Stack the inequality constraints
                        Ain = np.r_[Ain, c_Ain]
                        bin = np.r_[bin, c_bin]

        # Lineaself.robot.component of objective function: the manipulability Jacobian
        c = np.concatenate(
            (
                np.zeros(2),
                -self.robot.jacobm(start=self.robot.links[4]).reshape((self.n - 2,)),
                np.zeros(6),
            )
        )

        if self.config.home_cost:
            c1 = np.concatenate(
                (
                    np.zeros(2),
                    self.robot.jacob_inf(
                        self.robot._q[2:], [self.robot.qr[2:]], 5.0
                    ).reshape((self.n - 2,)),
                    np.zeros(6),
                )
            )
            # lamda_2 = 1.0 * (self.robot.qd[1] /(self.robot.qdlim[1] / 2) )
            lamda_2 = 0.5
            # TODO NORMALIZE WITH MAX Vx
            c = c + lamda_2 * c1

        # Get base to face end-effector
        kε = 0.5
        bTe = self.robot.fkine(self.robot._q, include_base=False).A
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[1] = -ε

        # c[: self.n] -= self.config.acceleration_gain * self.last_qd

        # The loweself.robot.and uppeself.robot.bounds on the joint velocity and slack vaself.robot.able
        lb = -np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]

        # Solve foself.robot.the joint velocities dq
        self.qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
        failed = False
        if self.qd is None:
            print("failed")
            self.qd = np.zeros(self.n)
            failed = True
            return False, self.qd, failed
        self.qd = self.qd[: self.n]

        # if et > 0.5:
        #     self.qd *= 0.7 / et
        # else:
        #     self.qd *= 1.4
        # self.log(self.qd, collisions)

        if et < self.config.precision:
            return True, self.qd, False

        else:
            return False, self.qd, False

class MMNeoSDF:
    def __init__(
        self,
        robot: NeuralRobot,
        sdf_model: SDF,
        config: NeoConfig = NeoConfig(),
        logger: Optional[Logger] = None,
        mesh: Optional[sg.Mesh] = None,
        gt_obstacles: Optional[List[sg.CollisionShape]] = None,
        gt_robot: Optional[NeuralRobot] = None,
        env: Optional[swift.Swift] = None,
    ) -> None:
        self.robot = robot
        self.n = robot.n
        self.sdf_model = sdf_model
        self.config = config
        self.logger = logger
        self.gt_obstacles = gt_obstacles
        self.gt_robot = gt_robot
        self.env = env
        self.last_qd = np.zeros(self.n)
        self.debug_qd = np.zeros(self.n)
        if self.config.rviz:
            self.viz = PointsRvizVisualizer(mesh)

    def get_distance(self) -> torch.Tensor:
        _, distance, _ = self.robot.get_distance(self.sdf_model)

        return distance

    def step(self, Tep) -> tuple[bool, np.ndarray, bool]:
        """
        Compute velocity
        Returns
        -------
        reached: bool
            end effector reached the target position
        qd: np.ndarray
            Joint velocity of the arm
        failed: bool
            Flag if qp didnt found a solution
        """
        total_ti = time.perf_counter()

        T_We = self.robot.fkine(self.robot.q)
        T_eEp = sm.SE3(np.linalg.inv(Tep.A) @ T_We.A, check=False).norm()
        # pdb.set_trace()
        T_be = self.robot.fkine(
            self.robot.q,
            start=self.robot.links[self.robot.base_dofs + 1],
            include_base=False,
        )
        T_bep = (
            self.robot.fkine(
                self.robot.q,
                include_base=True,
                end=self.robot.links[self.robot.base_dofs],
            ).inv()
            * Tep
        )

        self.robot.transform[:3, :3] = torch.tensor(
            self.robot.base.R.T, device="cuda", dtype=torch.float32
        )
        # spatial error
        et = np.sum(np.abs(T_eEp.t)) + 1e-10

        # et = np.sum(np.abs(np.r_[T_eEp.t, T_eEp.rpy() * np.pi / 180]))
        # print("error:", et)
        # Gain term (labda) for control minimisation
        Y = self.config.lamda_q

        # Quadratic component of objective function
        Q = np.eye(self.robot.n + 6)

        # Joint velocity component of Q
        Q[: self.robot.n, : self.robot.n] *= Y

        if (
            self.config.fixed_slack == Slack.normal
            or self.config.fixed_slack == Slack.rpy
            or self.config.fixed_slack == Slack.free_rot
        ):
            Q[: self.robot.base_dofs, : self.robot.base_dofs] *= 1.0 / (et)
            et_for_slack = 1 / (et * 1)
        elif self.config.fixed_slack == Slack.fixed_free:
            # Q[:self.robot.base_dofs,:self.robot.base_dofs] *= 1.0
            et_for_slack = self.config.fixed_slack_value_free

        else:
            et_for_slack = self.config.fixed_slack_value_constraint

        # Slack component of Q
        Q[self.n :, self.n :] = (et_for_slack) * np.eye(6)

        if self.config.fixed_slack == Slack.free_rot:
            # Q[-1:, -1:] = 1e-3 * Q[-1:, -1:]  # REMOVE THIS LATER IS FOR TRYINHG
            Q[-4, -4] = 1 / 2 * Q[-4, -4]  # Add freedom on z
            v_b, _ = rtb.p_servo(T_be, T_bep, self.config.beta, method="angle-axis")
            norm_ev = np.linalg.norm(v_b[:3])
            if norm_ev > self.config.max_ev:
                # only cap the translational component
                v_b[:] = (self.config.max_ev / norm_ev) * v_b[:]
            v = v_b
            Aeq = np.c_[self.robot.jacob0(self.robot.q), np.eye(6)]
        elif self.config.fixed_slack == Slack.rpy:
            v_b, _ = rtb.p_servo(T_be, T_bep, self.config.beta, method="angle-axis")
            v = v_b
            Aeq = np.c_[self.robot.jacob0(self.robot.q), np.eye(6)]
        else:
            # Pose difference
            # e = np.empty(6)
            # Translational error
            # e[:3] = T_eEp[:3, -1]
            # # Angular error
            # e[3:] = sm.base.tr2rpy(T_eEp, unit="rad", order="zyx", check=False)
            v_e, _ = rtb.p_servo(
                T_We,
                Tep,
                self.config.beta,
                method="rpy",
                threshold=self.config.precision,
            )
            # print("v_e",
            norm_ev = np.linalg.norm(v_e[:3])
            if norm_ev > self.config.max_ev:
                # only cap the translational component
                # v_e[:3] = (self.config.max_ev / norm_ev) * v_e[:3]
                v_e[:] = (self.config.max_ev / norm_ev) * v_e[:]

                # print("v_e after", v_e)
            v = v_e
            Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
            #

        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = self.config.ps

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = self.config.pi
        #


        if self.config.collisions or self.config.collision_cost != "":
            foward_points_ti = time.perf_counter()
            X_WSp = self.robot.transform_points()
            foward_points_time = time.perf_counter() - foward_points_ti
            forward_old_time = 0.0
            model_ti = time.perf_counter()
            gradient, distance = self.sdf_model.get_closest(X_WSp)

            gradient = gradient.detach()
            distance = distance.detach()
            distance -= self.robot.SpheresRadius
            model_time = time.perf_counter() - model_ti

        inequalities_ti = time.perf_counter()
        Ain[: self.n, : self.n], bin[: self.n] = self.robot.joint_velocity_damper(
            ps, pi, self.n
        )

        # if self.config.acc_constraint:
        #     acc_Ain = np.zeros((self.n * 2 + 6, self.n + 6))
        #     acc_bin = np.zeros(self.n * 2 + 6)
        #     for i in range(self.n):
        #         acc_Ain[i, i] = 1
        #         acc_Ain[i + self.n, i] = -1
        #         acc_bin[i] = self.config.max_acceleration_step[i] + self.last_qd[i]
        #         acc_bin[i + self.n] = (
        #             self.config.max_acceleration_step[i] - self.last_qd[i]
        #         )
        #     Ain = np.r_[Ain, acc_Ain]
        #     bin = np.r_[bin, acc_bin]

        if self.config.gt_collisions and self.gt_obstacles is not None:
            # pdb.set_trace()
            c_Ain = np.empty((0, self.n))
            c_bin = np.empty((0))
            for collision in self.gt_obstacles:
                # Form the velocity damper inequality contraint for each collision
                # object on the robot to the collision in the scene
                _c_Ain, _c_bin = self.robot.link_collision_damper(
                    collision,
                    self.robot.q[:],
                    di=self.config.di,
                    ds=self.config.ds,
                    xi=self.config.xi,
                )

                # If there are any parts of the robot within the influence distance
                # to the collision in the scene
                if _c_Ain is not None and _c_bin is not None:
                    _c_Ain = _c_Ain[:, : self.n]

                    # Stack the inequality constraints
                    c_Ain = np.r_[c_Ain, _c_Ain]
                    c_bin = np.r_[c_bin, _c_bin]

            distances = rtb_utils.get_distances(self.robot, self.gt_obstacles)
            min_distance = min(distances.values())
            print("min_distance", min_distance)


        elif self.config.collisions or self.config.collision_cost:
            if self.robot.spheres:
                c_Ain, c_bin  = compute_collision_constraints(self.robot, gradient, distance, self.config)
            else:

                c_Ain, c_bin, indices = self.robot.link_collision_constraint_torch(
                    gradient,
                    distance,
                    di=self.config.di,
                    ds=self.config.ds,
                    xi=self.config.xi,
                    aprox=self.config.approx_jacobian,
                    only_min=self.config.only_min,
                    top_k=self.config.topk,
                )
            c_Ain = c_Ain.detach().cpu().numpy()
            c_bin = c_bin.detach().cpu().numpy()

            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]
        # c_Ain = None
        torch.cuda.synchronize()
        inequalities_time = time.perf_counter() - inequalities_ti


        if self.config.collisions or self.config.gt_collisions:
            if c_Ain is not None and c_bin is not None:
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

        # component of objective function: the manipulability Jacobian
        if self.config.manipulability == Manipulability.active:
            c = np.concatenate(
                (
                    np.zeros(self.robot.base_dofs),
                    -self.robot.jacobm(
                        start=self.robot.links[self.robot.base_dofs + 2]
                    ).reshape((self.n - self.robot.base_dofs,)),
                    np.zeros(6),
                )
            )
        elif self.config.manipulability == Manipulability.opposite:
            c = np.concatenate(
                (
                    np.zeros(self.robot.base_dofs),
                    self.robot.jacobm(
                        start=self.robot.links[self.robot.base_dofs + 2]
                    ).reshape((self.n - self.robot.base_dofs,)),
                    np.zeros(6),
                )
            )
        else:
            c = np.concatenate(
                (np.zeros(self.robot.base_dofs), np.zeros(7), np.zeros(6))
            )

        c_mani = c.copy()
        collision_lamda = 1.0
        if self.config.collision_cost != "":
            if c_Ain.shape[0] != 0:
                if self.config.collision_cost == "min":
                    min_distance = np.argmin(c_bin)
                    collision = c_Ain[min_distance, :]
                elif self.config.collision_cost == "avg":
                    collision = np.average(c_Ain, axis=0)
                elif self.config.collision_cost == "w_avg":
                    weights = 1 - c_bin
                    collision = np.average(c_Ain, axis=0, weights=weights)
                elif self.config.collision_cost == "w2_avg":
                    weights = 1 - c_bin**2
                    collision = np.average(c_Ain, axis=0, weights=weights)
                elif self.config.collision_cost == "w3_avg":
                    weights = 1 - c_bin**2
                    collision = np.average(c_Ain, axis=0, weights=weights**2)
                else:
                    collision = np.zeros(self.robot.n + 6)
            else:
                collision = np.zeros(self.robot.n + 6)
            if self.config.gt_collisions:
                x = min_distance
            else:
                x = torch.min(distance).detach().cpu().item()
            # TODO review this
            di = self.config.di
            ds = self.config.ds
            collision_lamda = (self.config.collision_gain / (di - ds) ** 2) * (
                x - di
            ) ** 2
            # collision_lamda = self.config.collision_gain / (di - ds) * (di - x)
            # print("collision lamda", collision_lamda)
            c = c + collision_lamda * collision

        if self.config.home_cost:
            home_pos = np.array([0.131, -1.43, -0, -2.796, -0.023, 1.42, 0.856])
            c1 = np.concatenate(
                (
                    np.zeros(2),
                    self.robot.jacob_inf(self.robot.q[2:], [home_pos], 2).reshape(
                        (self.n - 2,)
                    ),
                    np.zeros(6),
                )
            )
            print(c1)
            lamda_2 = np.abs(self.last_qd[0]) * 0.25
            c = c + lamda_2 * c1

        # Get base to face end-effector
        if self.config.orientation_cost:
            kε = 0.5
            bTe = self.robot.fkine(self.robot.q, include_base=False).A
            θε = math.atan2(bTe[1, -1], bTe[0, -1])
            # if np.abs(θε) < 0.1:
            #     #     # pass
            #     print("to low", θε)
            # else:
            ε = kε * θε
            c[self.robot.base_dofs - 1] = c[self.robot.base_dofs - 1] - ε

        lb = -1 * np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]
        # c[: self.n] -= self.config.acceleration_gain * self.last_qd

        solver_ti = time.perf_counter()
        # Solve foself.robot.the joint velocities dq
        # remove row with zeros in Ain
        mask = np.all(Ain == 0, axis=1)
        Ain = Ain[~mask]
        bin = bin[~mask]
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
        solver_time = time.perf_counter() - solver_ti
        failed = False
        if qd is not None:
            oqd = qd.copy()
            qd = qd[: self.n]
        else:
            qd = np.zeros(self.n)
            oqd = np.zeros(self.n + 6)
            # print(failed)
            failed = True
        self.debug_qd = qd

        if self.config.real_robot:
            qd *= self.config.vel_scaler
            # pdb.set_trace()

            alpha = 0.9
            qd = alpha * qd + (1 - alpha) * self.last_qd
            # qd = np.clip(
            #     qd,
            #     self.last_qd - self.config.max_acceleration_step,
            #     self.last_qd + self.config.max_acceleration_step,
            # )

        self.last_qd = qd

        # pdb.set_trace()
        # mani_gain = c_mani @ oqd
        # collision_gain = collision @ qd
        total_time = time.perf_counter() - total_ti
        if self.config.log and self.logger is not None:
            # save_process = mp.Process(target=self.log(qd, distance))
            mani_gain = c_mani @ oqd
            # save_process.start()
            log = {}
            vlog = {}
            log["qd"] = qd[: self.robot.n]
            log["q"] = np.copy(self.robot.q)
            if self.gt_robot is not None:
                # gt_X_WSp = self.gt_robot.transform_points()
                # _, gt_distance = self.sdf_model.get_closest(gt_X_WSp)
                _, gt_distance, _ = self.gt_robot.get_distance(self.sdf_model)
                log["gt_distance"] = self.gt_robot.get_distance_links(gt_distance)
                # if (gt_distance.min()) < 0.0:
                #     print("GT collision STOPPING EXPERIMENT" * 5)
                #     exit(1)
                log["mean_gt_distance"] = float(
                    torch.mean(gt_distance).detach().cpu().item()
                )
            if self.config.collisions:
                log["distance"] = self.robot.get_distance_links(distance)
                log["mean_distance"] = float(torch.mean(distance).detach().cpu().item())
            if self.config.gt_collisions and self.config.collisions:
                pb_distance = rtb_utils.get_distances(self.robot, self.gt_obstacles)
                log["distance"] = pb_distance
                log["mean_distance"] = np.mean(list(pb_distance.values()))
                # log["mean_distance"] = float(np.mean(distance))

            log["eef_pose"] = T_We

            log["base"] = self.robot.base
            log["error"] = et
            log["slack_cost"] = et_for_slack
            log["foward_points_time"] = foward_points_time
            log["forward_old_time"] = forward_old_time
            log["model_time"] = model_time
            log["inequalities_time"] = inequalities_time
            log["solver_time"] = solver_time
            log["total_time"] = total_time
            log["other_time"] = total_time - (
                foward_points_time + model_time + inequalities_time + solver_time
            )
            # vlog["bin"] = bin
            # vlog["Ain"] = Ain

            if self.config.home_cost:
                pass
            if self.config.collision_cost:
                # pdb.set_trace()
                log["collision_gain"] = collision @ oqd
                log["collision"] = collision
            log["mani_gain"] = mani_gain
            self.logger.log(log, vlog)
        if self.config.collisions:
            self.distance = distance
        if et < self.config.precision:
            return True, qd, failed

        else:
            return False, qd, failed

class qp_base:
    def __init__(self, robot: NeuralRobot, sdf_model: SDF) -> None:
        """QP controller for the base of the robot

        Parameters
        ----------
        robot : NeuralRobot
            Robot to control, we use a neural robot to have the sampling points
        sdf_model : SDF
            Trained SDF model for performing collisions queries with the environment
        """
        self.robot = robot
        self.n = robot.n
        self.sdf_model = sdf_model

    def step(self, Tep: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Perform a step of the controller to move the robot towards the target pose

        Parameters
        ----------
        Tep : np.ndarray
            Target pose for the end effector

        Returns
        -------
        Tuple[bool, np.ndarray]
            Tuple of a boolean indicating if the robot has arrived and the joint velocities
        """
        T_We = self.robot.fkine(self.robot.q)

        T_eEp = np.linalg.inv(Tep) @ T_We.A

        # spatial error
        et = np.sum(np.abs(T_eEp[0:3, 3]))

        # Gain term (labda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.robot.n + 6)
        #

        # Joint velocity component of Q
        Q[: self.robot.n, : self.robot.n] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[self.n :, self.n :] = (1.0 / et) * np.eye(6)

        v, _ = rtb.p_servo(T_We, Tep, 1.5)

        v[-1] = np.arctan2(v[1], v[0])

        # v[3:] *= 2.3 #1.3
        print(v)

        # The equality contraints
        Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.1

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Get the distance and gradients:
        X_WSp = self.robot.transform_points()
        gradient, distance = self.sdf_model.get_closest(X_WSp)

        c_Ain, c_bin = self.robot.link_collision_constraint_torch(
            gradient, distance, di=2, ds=0.01, xi=1
        )

        # Move results back to numpy to integrate with the QP
        c_Ain = c_Ain.detach().cpu().numpy()
        c_bin = c_bin.detach().cpu().numpy()

        # Form the joint limit velocity damper
        Ain[: self.n, : self.n], bin[: self.n] = self.robot.joint_velocity_damper(
            ps, pi, self.n
        )

        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]



        c = np.zeros(self.robot.n + 6)

        # # get base to target angle
        kε = 0.5


        r, p, y = sm.SE3(T_eEp).rpy(order="xyz")
        θε = y
        ε = kε * θε
        c[5] = -ε

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
        if qd is None:
            qd = np.zeros(self.n)
            return False, qd
        qd = qd[: self.n]

        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4

        if et < 0.02:
            return True, qd

        else:
            return False, qd


class GripperController:
    def __init__(self, gripper) -> None:
        self.gripper = gripper

    def step(self, q):
        qd = 0.01 * (q - self.gripper.q[0])

        error = np.linalg.norm(qd)

        if error < 0.01:
            return True, qd
        else:
            return False, qd


class VelocityController:
    def __init__(self, robot) -> None:
        self.robot = robot

    def step(self, q):
        qd = self.robot.qdlim[2:] * (q - self.robot.q)
        qd = np.clip(qd, -self.robot.qdlim[2:] * 1, self.robot.qdlim[2:] * 1)
        error = np.linalg.norm(q - self.robot.q)
        if error < 0.1:
            return True, np.zeros(9)
        else:
            return False, qd


