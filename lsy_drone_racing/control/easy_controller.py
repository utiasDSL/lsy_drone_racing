"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

from lsy_drone_racing.control import Controller

from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
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

        
        def calc_waypoints(drone_init_pos, gates_pos, gates_quat, distance = 0.5, num_int_pnts = 5):
            gates_rotates = R.from_quat(gates_quat)
            rot_matrices = np.array(gates_rotates.as_matrix())
            gates_norm = np.array(rot_matrices[:,:,1])
            num_gates = gates_pos.shape[0]

            front = gates_pos - distance * gates_norm
            middle = gates_pos
            back = gates_pos + distance * gates_norm
            # wp = np.concatenate([front, middle, back], axis=1).reshape(num_gates, 3, 3).reshape(-1,3)
            wp = np.concatenate([gates_pos - distance * gates_norm + i/num_int_pnts * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
            wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)
            # print(gates_pos)
            # print(gates_norm)
            # print(np.concatenate([front, back], axis=1).reshape(num_gates, 2, 3).reshape(-1,3))

            return wp
        
        self.t_total = 30
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

        def avoid_collision(waypoints, obstacles_pos, safe_dist):
            t = np.linspace(0, self.t_total, len(waypoints))
            trajectory = CubicSpline(t, waypoints)
            t_axis = np.linspace(0, self.t_total, self._freq * self.t_total)
            wp = trajectory(t_axis)

            for obst in obstacles_pos:
                flag = False
                t_results = []
                wp_results = []
                for i in range(wp.shape[0]):
                    point = wp[i]
                    if np.linalg.norm(obst[:2] - point[:2]) < safe_dist and not flag: # first time visit
                        # map it to new point
                        flag = True
                        in_idx = i
                    elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist and flag:    # visited and out
                        out_idx = i
                        flag = False
                        direction = wp[in_idx][:2] - obst[:2] + wp[out_idx][:2] - obst[:2]
                        direction = direction / np.linalg.norm(direction)
                        new_point_xy = obst[:2] + direction * safe_dist
                        new_point_z = (wp[in_idx][2] + wp[out_idx][2])/2
                        new_point = np.concatenate([new_point_xy, [new_point_z]])
                        t_results.append((t_axis[in_idx] + t_axis[out_idx])/2)
                        wp_results.append(new_point)
                    elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist:   # out
                        t_results.append(t_axis[i])
                        wp_results.append(point)
                t_axis = np.array(t_results)
                wp = np.array(wp_results)


            return t_axis, wp
        

        waypoints = calc_waypoints(obs['pos'], obs['gates_pos'], obs['gates_quat'])
        t, waypoints = avoid_collision(waypoints, obs['obstacles_pos'], 0.2)
        self.trajectory = CubicSpline(t, waypoints)


        dt = np.linspace(0, self.t_total, 50 * len(waypoints))
        waypoints = np.array([self.trajectory(tau) for tau in dt])

        import matplotlib.pyplot as plt

        # t = np.linspace(0, self.t_total, len(waypoints))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
        # Create 3D graph!
        
        # Draw path
        ax.plot(x, y, z, marker='o', linestyle='-', color='b')

        # Set axes
        ax.set_title("3D Waypoints Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(True)
        plt.tight_layout()
        plt.show()
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
        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

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
