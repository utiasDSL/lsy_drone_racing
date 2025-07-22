"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


PARAMS_RPY = np.array([[-12.7, 10.15], [-12.7, 10.15], [-8.117, 14.36]])
PARAMS_ACC = np.array([0.1906, 0.4903])
MASS = 0.027
GRAVITY = 9.81
THRUST_MIN = 0.02
THRUST_MAX = 0.1125


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    """Model setting"""
    # define basic variables in state and input vector
    pos = vertcat(MX.sym("x"), MX.sym("y"), MX.sym("z"))
    vel = vertcat(MX.sym("vx"), MX.sym("vy"), MX.sym("vz"))
    rpy = vertcat(MX.sym("r"), MX.sym("p"), MX.sym("y"))

    r_cmd, p_cmd, y_cmd = MX.sym("r_cmd"), MX.sym("p_cmd"), MX.sym("y_cmd")
    thrust_cmd = MX.sym("thrust_cmd")

    # define state and input vector
    states = vertcat(pos, vel, rpy)
    inputs = vertcat(r_cmd, p_cmd, y_cmd, thrust_cmd)

    # Define nonlinear system dynamics
    pos_dot = vel
    z_axis = vertcat(
        cos(rpy[0]) * sin(rpy[1]) * cos(rpy[2]) + sin(rpy[0]) * sin(rpy[2]),
        cos(rpy[0]) * sin(rpy[1]) * sin(rpy[2]) - sin(rpy[0]) * cos(rpy[2]),
        cos(rpy[0]) * cos(rpy[1]),
    )
    thrust = PARAMS_ACC[0] + PARAMS_ACC[1] * inputs[3]
    vel_dot = thrust * z_axis / MASS - np.array([0.0, 0.0, GRAVITY])
    rpy_dot = PARAMS_RPY[:, 0] * rpy + PARAMS_RPY[:, 1] * inputs[:3]
    f = vertcat(pos_dot, vel_dot, rpy_dot)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs

    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights (we only give pos reference anyway)
    Q = np.diag(
        [
            10.0,  # pos
            10.0,  # pos
            10.0,  # pos
            0.0,  # vel
            0.0,  # vel
            0.0,  # vel
            0.0,  # rpy
            0.0,  # rpy
            0.0,  # rpy
        ]
    )

    R = np.diag(
        [
            5.0,  # rpy
            5.0,  # rpy
            5.0,  # rpy
            8.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:3, :3] = np.eye(3)  # Only select position states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:3, :3] = np.eye(3)  # Only select position states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 60°)
    ocp.constraints.lbx = np.array([-1.0, -1.0, -1.0])
    ocp.constraints.ubx = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbx = np.array([6, 7, 8])

    # Set Input Constraints (rpy < 60°)
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, THRUST_MIN * 4])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, THRUST_MAX * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="c_generated_code/lsy_example_mpc.json", verbose=verbose
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 30
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )

        des_completion_time = 8
        ts = np.linspace(0, des_completion_time, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        ts = np.linspace(0, des_completion_time, int(config.env.freq * des_completion_time))
        x_des = cs_x(ts)
        y_des = cs_y(ts)
        z_des = cs_z(ts)

        x_des = np.concatenate((x_des, [x_des[-1]] * self._N))
        y_des = np.concatenate((y_des, [y_des[-1]] * self._N))
        z_des = np.concatenate((z_des, [z_des[-1]] * self._N))
        self._waypoints_pos = np.stack((x_des, y_des, z_des)).T
        self._waypoints_yaw = x_des * 0

        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(x_des) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        x0 = np.concatenate((obs["pos"], obs["vel"], obs["rpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i + self._N]  # position
        yref[:, 5] = self._waypoints_yaw[i + self._N]  # yaw
        yref[:, 9] = MASS * GRAVITY  # hover thrust
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        # WARNING: The following line is only for legacy reason!
        # The Crazyflie uses the rpyt command format, the environment
        # take trpy format. Remove this line as soon as the env
        # also works with rpyt!
        u0 = np.array([u0[3], *u0[:3]], dtype=np.float32)

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
