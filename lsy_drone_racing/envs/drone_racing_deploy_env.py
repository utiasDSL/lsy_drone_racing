"""Deployment environments for real-world drone racing.

This module provides environments for deploying drone racing algorithms on physical hardware,
mirroring the functionality of the simulation environments in the
:mod:`~lsy_drone_racing.envs.drone_racing_env` module.

Key components:

* :class:`~.DroneRacingDeployEnv`: A Gymnasium environment for controlling a real Crazyflie drone in
  a physical race track, using Vicon motion capture for positioning.
* :class:`~.DroneRacingThrustDeployEnv`: A variant of :class:`~.DroneRacingDeployEnv` that uses
  collective thrust and attitude commands for control.

These environments maintain consistent interfaces with their simulation counterparts
(:class:`~.DroneRacingEnv` and :class:`~.DroneRacingThrustEnv`), allowing for seamless transition
from simulation to real-world deployment. They handle the complexities of interfacing with physical
hardware while providing the same observation and action spaces as the simulation environments.

The module integrates with ROS, Crazyswarm, and Vicon systems to enable real-world drone control and
tracking in a racing scenario.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.closing_controller import ClosingController
from lsy_drone_racing.sim.drone import Drone
from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import check_gate_pass
from lsy_drone_racing.utils.import_utils import get_ros_package_path, pycrazyswarm
from lsy_drone_racing.utils.ros_utils import check_drone_start_pos, check_race_track
from lsy_drone_racing.vicon import Vicon

if TYPE_CHECKING:
    from munch import Munch
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingDeployEnv(gymnasium.Env):
    """A Gymnasium environment for deploying drone racing algorithms on real hardware.

    This environment mirrors the functionality of the
    class:~lsy_drone_racing.envs.drone_racing_env.DroneRacingEnv, but interfaces with real-world
    hardware (Crazyflie drone and Vicon motion capture system) instead of a simulation.

    Key features:
    - Interfaces with a Crazyflie drone for physical control
    - Uses Vicon motion capture for precise position tracking
    - Maintains the same observation and action spaces as the simulation environment
    - Tracks progress through gates in a real-world racing scenario
    - Provides safety checks for drone positioning and track setup

    The observation space and action space are the same as the
    class:~lsy_drone_racing.envs.drone_racing_env.DroneRacingEnv.

    This environment allows for a transition from simulation to real-world deployment, maintaining
    consistent interfaces and functionalities to avoid any code changes when moving from simulation
    to physical hardware.
    """

    CONTROLLER = "mellinger"

    def __init__(self, config: dict | Munch):
        """Initialize the crazyflie drone and the motion capture system.

        Args:
            config: The configuration of the environment.
        """
        super().__init__()
        self.config = config
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(13,))
        n_gates, n_obstacles = (
            len(config.env.track.get("gates")),
            len(config.env.track.get("obstacles")),
        )
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "target_gate": spaces.Discrete(n_gates, start=-1),
                "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
                "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(n_gates, 3)),
                "gates_in_range": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=np.bool_),
                "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
                "obstacles_in_range": spaces.Box(
                    low=0, high=1, shape=(n_obstacles,), dtype=np.bool_
                ),
            }
        )
        self.target_gate = 0
        crazyswarm_config_path = (
            get_ros_package_path("crazyswarm", heuristic_search=True) / "launch/crazyflies.yaml"
        )
        # pycrazyswarm expects strings, not Path objects, so we need to convert it first
        swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
        self.cf = swarm.allcfs.crazyflies[0]
        names = []
        if not self.config.deploy.practice_without_track_objects:
            names += [f"gate{g}" for g in range(1, len(config.env.track.gates) + 1)]
            names += [f"obstacle{g}" for g in range(1, len(config.env.track.obstacles) + 1)]
        self.vicon = Vicon(track_names=names, timeout=5)
        self.symbolic = None
        if config.env.symbolic:
            sim = Sim(
                track=config.env.track,
                sim_freq=config.sim.sim_freq,
                ctrl_freq=config.sim.ctrl_freq,
                disturbances=getattr(config.sim, "disturbances", {}),
                randomization=getattr(config.env, "randomization", {}),
                physics=config.sim.physics,
            )
            self.symbolic = sim.symbolic()
        self._last_pos = np.zeros(3)

        self.gates_visited = np.array([False] * len(config.env.track.gates))
        self.obstacles_visited = np.array([False] * len(config.env.track.obstacles))

        # Use internal variable to store results of self.obs that is updated every time
        # self.obs is invoked in order to prevent calling it more often than necessary.
        self._obs = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        We cannot reset the track in the real world. Instead, we check if the gates, obstacles and
        drone are positioned within tolerances.
        """
        if (
            self.config.deploy.check_race_track
            and not self.config.deploy.practice_without_track_objects
        ):
            check_race_track(self.config)
        if self.config.deploy.check_drone_start_pos:
            check_drone_start_pos(self.config)
        self._last_pos[:] = self.vicon.pos[self.vicon.drone_name]
        self.target_gate = 0
        info = self.info
        info["sim_freq"] = self.config.sim.sim_freq
        info["low_level_ctrl_freq"] = self.config.sim.ctrl_freq
        info["env_freq"] = self.config.env.freq
        info["drone_mass"] = 0.033  # Crazyflie 2.1 mass in kg
        return self.obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Take a step in the environment.

        Warning:
            Step does *not* wait for the remaining time if the step took less than the control
            period. This ensures that controllers with longer action compute times can still hit
            their target frequency. Furthermore, it implies that loops using step have to manage the
            frequency on their own, and need to update the current observation and info before
            computing the next action.
        """
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rate)
        current_pos = self.vicon.pos[self.vicon.drone_name]
        self.target_gate += self.gate_passed(current_pos, self._last_pos)
        self._last_pos[:] = current_pos
        if self.target_gate >= len(self.config.env.track.gates):
            self.target_gate = -1
        terminated = self.target_gate == -1
        return self.obs, -1.0, terminated, False, self.info

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        return_home = True  # makes the drone simulate the return to home after stopping

        if return_home:
            # This is done to run the closing controller at a different frequency than the controller before
            # Does not influence other code, since this part is already in closing!
            # WARNING: When changing the frequency, you must also change the current _step!!!
            freq_new = 100  # Hz
            self.config.env.freq = freq_new
            t_step_ctrl = 1 / self.config.env.freq

            obs = self.obs
            obs["acc"] = np.array(
                [0, 0, 0]
            )  # TODO, use actual value when avaiable or do one step to calculate from velocity
            info = self.info
            info["env_freq"] = self.config.env.freq
            info["drone_start_pos"] = self.config.env.track.drone.pos

            controller = ClosingController(obs, info)
            t_total = controller.t_total

            for i in np.arange(int(t_total / t_step_ctrl)):  # hover for some more time
                action = controller.compute_control(obs)
                action = action.astype(np.float64)  # Drone firmware expects float64
                pos, vel, acc, yaw, rpy_rate = (
                    action[:3],
                    action[3:6],
                    action[6:9],
                    action[9],
                    action[10:],
                )
                self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rate)
                obs = self.obs
                obs["acc"] = np.array([0, 0, 0])
                controller.step_callback(action, obs, 0, True, False, info)
                time.sleep(t_step_ctrl)

        self.cf.notifySetpointsStop()
        self.cf.land(0.05, 2.0)

    @property
    def obs(self) -> dict:
        """Return the observation of the environment."""
        drone = self.vicon.drone_name
        rpy = self.vicon.rpy[drone]
        ang_vel = R.from_euler("xyz", rpy).inv().apply(self.vicon.ang_vel[drone])
        obs = {
            "pos": self.vicon.pos[drone].astype(np.float32),
            "rpy": rpy.astype(np.float32),
            "vel": self.vicon.vel[drone].astype(np.float32),
            "ang_vel": ang_vel.astype(np.float32),
        }

        sensor_range = self.config.env.sensor_range
        n_gates = len(self.config.env.track.gates)

        obs["target_gate"] = self.target_gate if self.target_gate < n_gates else -1

        drone_pos = self.vicon.pos[self.vicon.drone_name]

        gates_pos = np.array([g.pos for g in self.config.env.track.gates])
        gates_rpy = np.array([g.rpy for g in self.config.env.track.gates])
        gate_names = [f"gate{g}" for g in range(1, len(gates_pos) + 1)]

        obstacles_pos = np.array([o.pos for o in self.config.env.track.obstacles])
        obstacle_names = [f"obstacle{g}" for g in range(1, len(obstacles_pos) + 1)]

        # Update objects position with vicon data if not in practice mode and object
        # either is in range or was in range previously.
        if not self.config.deploy.practice_without_track_objects:
            real_gates_pos = np.array([self.vicon.pos[g] for g in gate_names])
            real_gates_rpy = np.array([self.vicon.rpy[g] for g in gate_names])
            real_obstacles_pos = np.array([self.vicon.pos[o] for o in obstacle_names])

            # Use x-y distance to calucate sensor range, otherwise it would depend on the height of the drone
            # and obstacle how early the obstacle is in range.
            in_range = np.linalg.norm(real_gates_pos[:, :2] - drone_pos[:2], axis=1) < sensor_range
            self.gates_visited = np.logical_or(self.gates_visited, in_range)
            gates_pos[self.gates_visited] = real_gates_pos[self.gates_visited]
            gates_rpy[self.gates_visited] = real_gates_rpy[self.gates_visited]
            obs["gates_in_range"] = in_range

            in_range = (
                np.linalg.norm(real_obstacles_pos[:, :2] - drone_pos[:2], axis=1) < sensor_range
            )
            self.obstacles_visited = np.logical_or(self.obstacles_visited, in_range)
            obstacles_pos[self.obstacles_visited] = real_obstacles_pos[self.obstacles_visited]
            obs["obstacles_in_range"] = in_range

        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        self._obs = obs
        return obs

    @property
    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": [], "symbolic_model": self.symbolic}

    def gate_passed(self, pos: NDArray[np.floating], prev_pos: NDArray[np.floating]) -> bool:
        """Check if the drone has passed the current gate.

        Args:
            pos: Current drone position.
            prev_pos: Previous drone position.
        """
        n_gates = len(self.config.env.track.gates)
        if self.target_gate < n_gates and self.target_gate != -1:
            # Real gates measure 0.4m x 0.4m, we account for meas. error
            gate_size = (0.56, 0.56)
            gate_pos = self._obs["gates_pos"][self.target_gate]
            gate_rot = R.from_euler("xyz", self._obs["gates_rpy"][self.target_gate])
            return check_gate_pass(gate_pos, gate_rot, gate_size, pos, prev_pos)
        return False


class DroneRacingThrustDeployEnv(DroneRacingDeployEnv):
    """A Gymnasium environment for deploying drone racing algorithms on real hardware.

    This environment mirrors the functionality of the
    class:~lsy_drone_racing.envs.drone_racing_thrust_env.DroneRacingThrustEnv, but interfaces with
    real-world hardware (Crazyflie drone and Vicon motion capture system) instead of a simulation.
    """

    def __init__(self, config: dict | Munch):
        """Initialize the crazyflie drone and the motion capture system.

        Args:
            config: The configuration of the environment.
        """
        super().__init__(config)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        self.drone = Drone("mellinger")

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Take a step in the environment.

        Warning:
            Step does *not* wait for the remaining time if the step took less than the control
            period. This ensures that controllers with longer action compute times can still hit
            their target frequency. Furthermore, it implies that loops using step have to manage the
            frequency on their own, and need to update the current observation and info before
            computing the next action.
        """
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        collective_thrust, rpy = action[0], action[1:]
        rpy_deg = np.rad2deg(rpy)
        collective_thrust = self.drone._thrust_to_pwms(collective_thrust)
        self.cf.cmdVel(*rpy_deg, collective_thrust)
        current_pos = self.vicon.pos[self.vicon.drone_name]
        self.target_gate += self.gate_passed(current_pos, self._last_pos)
        self._last_pos[:] = current_pos
        if self.target_gate >= len(self.config.env.track.gates):
            self.target_gate = -1
        terminated = self.target_gate == -1
        return self.obs, -1.0, terminated, False, self.info
