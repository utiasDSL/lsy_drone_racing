"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosOcpSolver
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.acados_model import export_quadrotor_ode_model, setup_ocp
from lsy_drone_racing.control.collision_avoidance import CollisionAvoidanceHandler
from lsy_drone_racing.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

OBSTACLE_RADIUS = 0.15  # Radius of the obstacles in meters
GATE_LENGTH = 0.50  # Length of the gate in meters
ELLIPSOID_RADIUS = 0.15  # Diameter of the ellipsoid in meters
ELLIPSOID_LENGTH = 0.7  # Length of the ellipsoid in meters


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
        self.obs = obs

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
                obs["gates_pos"][2] + [0.1, 0.15, 0.4],
                obs["gates_pos"][2] + [0.1, 0.15, 0.8],
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

        # Setup collision avoidance
        num_gates = len(obs["gates_pos"])
        num_obstacles = len(obs["obstacles_pos"])
        self.collision_avoidance_handler = CollisionAvoidanceHandler(
            num_gates,
            num_obstacles,
            GATE_LENGTH,
            ELLIPSOID_LENGTH,
            ELLIPSOID_RADIUS,
            OBSTACLE_RADIUS,
        )

        # Setup the acados model and solver
        model = export_quadrotor_ode_model()
        self.collision_avoidance_handler.setup_model(model)
        ocp = setup_ocp(model, self.T_HORIZON, self.N)
        self.collision_avoidance_handler.setup_ocp(ocp)
        self.acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file="lsy_example_mpc.json", verbose=True
        )

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

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

        self.collision_avoidance_handler.update_parameters(self.acados_ocp_solver, self.N, obs)

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

        self.collision_avoidance_handler.draw_collision_bodies(env, self.obs)

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
