"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from asyncio.proactor_events import _ProactorBaseWritePipeTransport
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

from crazyflow.constants import GRAVITY, MASS
from stable_baselines3 import PPO
    

class RLController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)

        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        gates_rotates = R.from_quat(obs['gates_quat'])
        rot_matrices = np.array(gates_rotates.as_matrix())
        self.gates_norm = np.array(rot_matrices[:,:,1])
        self.gates_pos = obs['gates_pos']
        self.init_pos = obs['pos']
        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)

    # visualize trajectory
    def visualize_traj(
        self, gate_positions: NDArray[np.floating], gate_normals: NDArray[np.floating], obst_positions: NDArray[np.floating] = None,
        trajectory: CubicSpline = None, waypoints: NDArray[np.floating] = None, drone_pos: NDArray[np.floating] = None,
    ) -> None:
        """Visualize trajectory, gates, obstacles and drone position."""
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig = plt.figure(num=1, figsize=(10,10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        ax = self.ax
        ax.cla()

        # Draw path
        if waypoints is not None:
            x, y, z = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
            ax.plot(x, y, z, marker='.', linestyle='--', color='b')
        if trajectory is not None:
            dt = np.linspace(0, self.t_total, 100)
            traj = trajectory(dt)
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            ax.plot(x, y, z, marker='x', linestyle='-', color='orange')

        # Draw gates
        for pos, normal in zip(gate_positions, gate_normals):
            ax.quiver(pos[0], pos[1], pos[2],
                normal[0], normal[1], normal[2],
                length=0.5, color='green', linewidth=1)
            
        # Draw obstacles
        if obst_positions is not None:
            for obst in obst_positions:
                x,y,z = obst
                ax.plot([x, x], [y, y], [0, 1.4], color='grey', linewidth=4)

        # Draw drone
        if drone_pos is not None:
            ax.plot([drone_pos[0]], [drone_pos[1]], [drone_pos[2]], marker='x', markersize=20, color='black')

        # Set axes
        ax.set_title("Planned Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        plt.tight_layout()
        plt.draw()
        # plt.show()


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        action_exec = self.act_bias
        # Actor NN
        return action_exec

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1









        return self._finished
        """Reset the time step counter."""
        self._tick = 0
