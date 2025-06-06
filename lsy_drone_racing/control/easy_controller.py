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
from scipy.spatial.transform import Rotation as R
from traitlets import TraitType

from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.ext_tools import TrajectoryTool

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray
    

class EasyController(FresssackController):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, ros_tx_freq : np.floating = None):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config, ros_tx_freq = ros_tx_freq)
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        # waypoints = np.array(
        #     [
        #         [1.0, 1.5, 0.05],
        #         [0.8, 1.0, 0.2],
        #         [0.55, -0.3, 0.5],
        #         [0.2, -1.3, 0.65],
        #         [1.1, -0.85, 1.1],
        #         [0.2, 0.5, 0.65],
        #         [0.0, 1.2, 0.525],
        #         [0.0, 1.2, 1.1],
        #         [-0.5, 0.0, 1.1],
        #         [-0.5, -0.5, 1.1],
        #     ]
        # )

        self.t_total = 12
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        gates_rotates = R.from_quat(obs['gates_quat'])
        rot_matrices = np.array(gates_rotates.as_matrix())
        self.gates_norm = np.array(rot_matrices[:,:,1])
        self.gates_pos = obs['gates_pos']
        self.init_pos = obs['pos']

        waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
        # t = np.linspace(0, self.t_total, len(waypoints))
        t, waypoints = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
        self.trajectory = self.trajectory_generate(self.t_total, waypoints)

        # self.visualize_traj(self.gates_pos, self.gates_norm, obst_positions=obs['obstacles_pos'], trajectory=self.trajectory, waypoints=waypoints, drone_pos=obs['pos'])
    
    def calc_waypoints(
            self, drone_init_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_norm: NDArray[np.floating], distance: float = 0.5 , num_int_pnts: int = 5,
    ) -> NDArray[np.floating]:
        """Compute waypoints interpolated between gates."""
        num_gates = gates_pos.shape[0]
        wp = np.concatenate([gates_pos - distance * gates_norm + i/(num_int_pnts-1) * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
        wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)

        return wp
    
    def trajectory_generate(
        self, t_total: float, waypoints: NDArray[np.floating],
    ) -> CubicSpline:
        """Generate a cubic spline trajectory from waypoints."""
        diffs = np.diff(waypoints, axis=0)
        segment_length = np.linalg.norm(diffs, axis=1)
        arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
        t = arc_cum_length / arc_cum_length[-1] * t_total
        return CubicSpline(t, waypoints)
    
    def avoid_collision(
        self, waypoints: NDArray[np.floating], obstacles_pos: NDArray[np.floating], safe_dist: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Modify waypoints to avoid collision with obstacles."""
        trajectory = self.trajectory_generate(self.t_total, waypoints)
        t_axis = np.linspace(0, self.t_total, self._freq * self.t_total)
        wp = trajectory(t_axis)

        for obst_idx, obst in enumerate(obstacles_pos):
            flag = False
            t_results = []
            wp_results = []
            for i in range(wp.shape[0]):
                point = wp[i]
                if np.linalg.norm(obst[:2] - point[:2]) < safe_dist and not flag: # first time visit
                    flag = True
                    in_idx = i
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist and flag:    # visited and out
                    out_idx = i
                    flag = False
                    # map it to new point
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
    
    def add_drone_to_waypoints(
        self, waypoints: NDArray[np.floating], drone_pos: NDArray[np.floating], safe_dist: float, curr_theta: float = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Force new trajectory to pass through drone position."""
        trajectory = self.trajectory_generate(self.t_total, waypoints)
        trajectory = TrajectoryTool().arclength_reparameterize(trajectory)
        t_axis = trajectory.x
        wp = trajectory(t_axis)
        theta, obst = TrajectoryTool().find_nearest_waypoint(trajectory, drone_pos, curr_theta)
        flag = False
        t_results = []
        wp_results = []
        for i in range(wp.shape[0]):
            point = wp[i]
            if np.fabs(i*0.05 - theta) < safe_dist and not flag: # first time visit
                flag = True
                in_idx = i
            elif np.fabs(i*0.05 - theta) >= safe_dist and flag:    # visited and out
                out_idx = i
                flag = False
                # map it to new point
                new_point = drone_pos
                t_results.append((t_axis[in_idx] + t_axis[out_idx])/2)
                wp_results.append(new_point)
            elif np.fabs(i*0.05 - theta) >= safe_dist:   # out
                t_results.append(t_axis[i])
                wp_results.append(point)
        t_axis = np.array(t_results)
        wp = np.array(wp_results)

        return t_axis, wp
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

    def pos_change_detect(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect if position of any gate or obstacle was changed."""
        if not hasattr(self, 'last_gate_flags'):
            self.last_gate_flags = np.array(obs['gates_visited'], dtype=bool)
            self.last_obst_flags = np.array(obs['obstacles_visited'], dtype=bool)
            return False

        curr_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        curr_obst_flags = np.array(obs['obstacles_visited'], dtype=bool)

        gate_triggered = np.any((~self.last_gate_flags) & curr_gate_flags)
        obst_triggered = np.any((~self.last_obst_flags) & curr_obst_flags)

        self.last_gate_flags = curr_gate_flags
        self.last_obst_flags = curr_obst_flags

        return gate_triggered or obst_triggered

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
        print(obs["gates_visited"])

        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        if self.pos_change_detect(obs):
            gates_rotates = R.from_quat(obs['gates_quat'])
            rot_matrices = np.array(gates_rotates.as_matrix())
            self.gates_norm = np.array(rot_matrices[:,:,1])
            self.gates_pos = obs['gates_pos']
            # replan trajectory
            waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
            t, waypoints = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
            self.trajectory = self.trajectory_generate(self.t_total, waypoints)
            # self.visualize_traj(self.gates_pos, self.gates_norm, obst_positions=obs['obstacles_pos'], trajectory=self.trajectory, drone_pos=obs['pos'])
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def get_trajectory_function(self):
        return self.trajectory
    def get_trajectory_waypoints(self): # list of waypoint sync to self._tick
        t_axis = np.linspace(0, self.t_total, self._freq * self.t_total)
        wp = self.trajectory(t_axis)
        return wp
    def set_tick(self, tick):
        self._tick = tick

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
