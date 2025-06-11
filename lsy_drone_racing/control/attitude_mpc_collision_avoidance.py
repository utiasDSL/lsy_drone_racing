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
from casadi import MX, cos, diag, horzcat, mtimes, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import draw_cylinder, draw_ellipsoid, draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

OBSTACLE_RADIUS = 0.15  # Radius of the obstacles in meters
GATE_LENGTH = 0.50  # Length of the gate in meters
ELLIPSOID_DIAMETER = 0.3  # Diameter of the ellipsoid in meters
ELLIPSOID_LENGTH = 0.7  # Length of the ellipsoid in meters


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs

    return model


class MPController(Controller):
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
        self.freq = config.env.freq
        self._tick = 0

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                obs["pos"],
                obs["obstacles_pos"][0] + [0.2, 0.5, -0.7],
                obs["obstacles_pos"][0] + [0.2, -0.3, -0.7],
                obs["gates_pos"][0]
                + 0.5 * (obs["obstacles_pos"][0] - [0, 0, 0.6] - obs["gates_pos"][0]),
                obs["gates_pos"][0] + [0.1, 0.1, 0],
                obs["gates_pos"][0] + [-0.3, -0.2, 0],
                obs["obstacles_pos"][1] + [-0.3, -0.3, -0.7],
                obs["gates_pos"][1] + [-0.1, -0.2, 0],
                obs["gates_pos"][1],
                obs["gates_pos"][1] + [0.2, 0.5, 0],
                obs["obstacles_pos"][0] + [-0.3, 0, -0.7],
                obs["gates_pos"][2] + [0.2, -0.5, 0],
                obs["gates_pos"][2] + [0.1, 0, 0],
                obs["gates_pos"][2] + [0.1, 0.15, 0],
                obs["gates_pos"][2] + [0.1, 0.15, 1],
                obs["obstacles_pos"][3] + [0.4, 0.3, -0.2],
                obs["obstacles_pos"][3] + [0.4, 0, -0.2],
                obs["gates_pos"][3],
                obs["gates_pos"][3] + [0, -0.5, 0],
            ]
        )
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        des_completion_time = 15
        ts = np.linspace(0, 1, int(self.freq * des_completion_time))

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        self.N = 30
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        self.obs = obs
        self._create_ocp_solver()

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

    def _create_ocp_solver(self) -> None:
        """Create the OCP solver for the MPC."""
        # Create the OCP solver with the predefined model and settings
        self.ocp = AcadosOcp()

        # set model
        self.ocp.model = export_quadrotor_ode_model()

        # Get Dimensions
        nx = self.ocp.model.x.rows()
        nu = self.ocp.model.u.rows()
        ny = nx + nu
        ny_e = nx

        # Set horizon
        self.ocp.solver_options.N_horizon = self.N

        ## Set Cost
        # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

        # Cost Type
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        # Weights
        Q = np.diag(
            [
                3.0,
                3.0,
                3.0,  # Position
                0.01,
                0.01,
                0.01,  # Velocity
                0.1,
                0.1,
                0.1,  # rpy
                0.01,
                0.01,  # f_collective, f_collective_cmd
                0.01,
                0.01,
                0.01,
            ]
        )

        R = np.diag([0.01, 0.01, 0.01, 0.01])

        Q_e = Q.copy()

        self.ocp.cost.W = scipy.linalg.block_diag(Q, R)
        self.ocp.cost.W_e = Q_e

        Vx = np.zeros((ny, nx))
        Vx[:nx, :] = np.eye(nx)  # Only select position states
        self.ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
        self.ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states
        self.ocp.cost.Vx_e = Vx_e

        # Set initial references (we will overwrite these later on to make the controller track the traj.)
        self.ocp.cost.yref = np.array(
            [
                1.0,
                1.0,
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        self.ocp.cost.yref_e = np.array(
            [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
        )

        # Set State Constraints
        self.ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
        self.ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
        self.ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

        # Set Input Constraints
        # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
        # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
        # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # We have to set x0 even though we will overwrite it later on.
        self.ocp.constraints.x0 = np.zeros((nx))

        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        self.ocp.solver_options.tol = 1e-5
        self.ocp.solver_options.qp_solver_cond_N = self.N
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_solver_iter_max = 10
        self.ocp.solver_options.nlp_solver_max_iter = 50
        self.ocp.solver_options.tf = self.T_HORIZON

        self._create_collisions(self.obs)

        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp, json_file="lsy_example_mpc.json", verbose=True
        )

    def _create_collisions(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Model the collisions in the environment as nonlinear constraints in the MPC."""
        drone_pos = self.ocp.model.x[:3]  # Get the drone position from the model state

        obs_params, obs_h_expr = self._create_obstacles(drone_pos, len(obs["obstacles_pos"]))
        gate_params, gate_h_expr = self._create_gate_collisions(drone_pos, len(obs["gates_pos"]))

        self.ocp.model.p = vertcat(*obs_params, *gate_params)  # Add obstacle position as parameter
        self.ocp.model.con_h_expr = vertcat(
            *obs_h_expr, *gate_h_expr
        )  # Add the nonlinear constraints

        # Set nonlinear constraints
        num_constraints = self.ocp.model.con_h_expr.rows()
        self.ocp.dims.nsh = num_constraints
        self.ocp.constraints.lh = np.array([0] * num_constraints)
        self.ocp.constraints.uh = np.array([1e10] * num_constraints)
        self.ocp.constraints.idxsh = np.array(list(range(num_constraints)))
        self.ocp.constraints.lsh = np.array([0] * num_constraints)
        self.ocp.constraints.ush = np.array([1e10] * num_constraints)
        self.ocp.cost.Zl = np.array([100] * num_constraints)  # penalty on slack (lower side)
        self.ocp.cost.Zu = np.array([0] * num_constraints)  # penalty on slack (upper side)
        self.ocp.cost.zl = np.array([100] * num_constraints)  # desired slack is zero
        self.ocp.cost.zu = np.array([0] * num_constraints)
        self.ocp.parameter_values = np.array([0.0] * self.ocp.model.p.rows())

    def _create_gate_collisions(self, drone_pos: MX, num_gates: int) -> tuple[MX, MX]:
        params = []
        h_expr = []

        ellipsoid_midpoints = np.array(
            [
                [GATE_LENGTH / 2, 0, 0],
                [-GATE_LENGTH / 2, 0, 0],
                [0, 0, GATE_LENGTH / 2],
                [0, 0, -GATE_LENGTH / 2],
            ]
        )
        ellipsoid_axes = np.array(
            [
                [ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_LENGTH / 2],
                [ELLIPSOID_LENGTH / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2],
                [ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_LENGTH / 2],
                [ELLIPSOID_LENGTH / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2],
            ]
        )

        for i in range(num_gates):
            gate_pos = MX.sym(f"p_gate{i}", 4)  # Gate position (x, y, z, yaw)
            params.append(gate_pos)

            center = gate_pos[:3]  # Center of the gate
            yaw = gate_pos[3]  # Yaw angle of the gate
            # Rotation matrix for yaw (around z)
            Rz = vertcat(
                horzcat(cos(yaw), -sin(yaw), 0), horzcat(sin(yaw), cos(yaw), 0), horzcat(0, 0, 1)
            )

            for midpoint, axes in zip(ellipsoid_midpoints, ellipsoid_axes):
                midpoint_vec = MX(midpoint)
                ellipsoid_center = center + mtimes(Rz, midpoint_vec)
                a, b, c = axes
                dpos = drone_pos - ellipsoid_center
                ellipsoid_val = (
                    dpos.T @ Rz.T @ diag([1 / a**2, 1 / b**2, 1 / c**2]) @ Rz @ dpos
                ) - 1
                h_expr.append(ellipsoid_val)

        return params, h_expr

    def _create_obstacles(self, drone_pos: MX, num_obstacles: int) -> tuple[list[MX], list[MX]]:
        """Model the obstacles in the environment. as nonlinear constraints in the MPC."""
        params = []
        h_expr = []
        for i in range(num_obstacles):
            center_obs = MX.sym(f"p_obs{i}", 2)
            params.append(center_obs)

            # Infinitly high cylinder around the obstacle with radius OBSTACLE_RADIUS
            pos_xy = drone_pos[:2]  # Get the x and y position of the drone
            con_h_expr = (pos_xy - center_obs).T @ (pos_xy - center_obs) - OBSTACLE_RADIUS**2
            h_expr.append(con_h_expr)
        return params, h_expr

    def _update_parameters(self, obs: dict[str, NDArray[np.floating]]) -> None:
        obstacle_positions = obs["obstacles_pos"]
        obstacle_params = obstacle_positions[:, :2].flatten()

        gate_positions = obs["gates_pos"]
        gate_rotations = R.from_quat(obs["gates_quat"]).as_euler("xyz", degrees=False)
        gate_params = np.hstack((gate_positions, gate_rotations[:, 2:])).flatten()
        params = np.concatenate((obstacle_params, gate_params))

        for i in range(self.N):
            gate_params = gate_positions[:, :4].flatten()
            self.acados_ocp_solver.set(i, "p", params)

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
        self.obs = obs
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        for j in range(self.N):
            yref = np.array(
                [
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                    0.35,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", yref)

        yref_N = np.array(
            [
                self.x_des[i + self.N - 1],
                self.y_des[i + self.N - 1],
                self.z_des[i + self.N - 1],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

        self._update_parameters(obs)

        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]

        return cmd

    def draw(self, env: DroneRaceEnv):
        """Draw the trajectory in the environment.

        Args:
            env (DroneRaceEnv): Environment to draw the trajectory in.
        """
        positions = []
        for i in range(self.N + 1):  # +1 to include terminal state
            x_pred = self.acados_ocp_solver.get(i, "x")
            pos = x_pred[:3]  # [x, y, z]
            positions.append(pos)
        draw_line(
            env,
            np.array(positions),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
            min_size=0.01,
            max_size=0.01,
        )

        obstacle_positions = self.obs["obstacles_pos"]
        obstacle_positions[:, 2] = 0  # Only use x and y positions
        for obstacle_pos in obstacle_positions:
            draw_cylinder(
                env,
                pos=obstacle_pos,
                size=np.array([OBSTACLE_RADIUS, 2.0]),
                rgba=np.array([0.2, 0.2, 0.8, 0.5]),
            )

        gate_positions = self.obs["gates_pos"]
        gate_quats = self.obs["gates_quat"]

        from scipy.spatial.transform import Rotation as R

        num_gates = gate_positions.shape[0]
        ellipsoid_midpoints = np.array(
            [
                [GATE_LENGTH / 2, 0, 0],
                [0, 0, GATE_LENGTH / 2],
                [-GATE_LENGTH / 2, 0, 0],
                [0, 0, -GATE_LENGTH / 2],
            ]
        )
        ellipsoid_axes = np.array(
            [
                [ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_LENGTH / 2],
                [ELLIPSOID_LENGTH / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2],
                [ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_LENGTH / 2],
                [ELLIPSOID_LENGTH / 2, ELLIPSOID_DIAMETER / 2, ELLIPSOID_DIAMETER / 2],
            ]
        )

        gate_yaws = R.from_quat(gate_quats).as_euler("xyz")[:, 2]
        for gate_idx in range(num_gates):
            gate_pos = gate_positions[gate_idx]
            yaw = gate_yaws[gate_idx]
            # Rotation matrix for yaw (around z)
            c = np.cos(yaw)
            s = np.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for midpoint, axes in zip(ellipsoid_midpoints, ellipsoid_axes):
                ellipsoid_center = gate_pos + Rz @ midpoint
                draw_ellipsoid(
                    env,
                    pos=ellipsoid_center,
                    size=axes,
                    rot=Rz,
                    rgba=np.array([0.8, 0.2, 0.2, 0.5]),
                )

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

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
