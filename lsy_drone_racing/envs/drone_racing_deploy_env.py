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
import rospy
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.sim import Sim
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
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )
        self.target_gate = 0
        crazyswarm_config_path = (
            get_ros_package_path("crazyswarm", heuristic_search=True) / "launch/crazyflies.yaml"
        )
        # pycrazyswarm expects strings, not Path objects, so we need to convert it first
        swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
        self.cf = swarm.allcfs.crazyflies[0]
        names = [f"gate{g}" for g in range(1, len(config.env.track.gates) + 1)]
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

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        We cannot reset the track in the real world. Instead, we check if the gates, obstacles and
        drone are positioned within tolerances.
        """
        check_race_track(self.config)
        check_drone_start_pos(self.config)
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

        Note:
            Sleeps for the remaining time if the step took less than the control period. This
            ensures that the environment is running at the correct frequency during deployment.
        """
        tstart = time.perf_counter()
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rate)
        if (dt := time.perf_counter() - tstart) < 1 / self.config.env.freq:
            rospy.sleep(1 / self.config.env.freq - dt)
        return self.obs, -1.0, False, False, self.info

    def close(self):
        """Close the environment by stopping the drone and landing."""
        self.cf.notifySetpointsStop()
        self.cf.land(0.02, 3.0)

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
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        drone_pos = self.vicon.pos[self.vicon.drone_name]
        gates_pos = np.array([g.pos for g in self.config.env.track.gates])
        gate_names = [f"gate{g}" for g in range(1, len(gates_pos) + 1)]
        real_gates_pos = np.array([self.vicon.pos[g] for g in gate_names])
        in_range = np.linalg.norm(real_gates_pos - drone_pos, axis=1) < sensor_range
        gates_pos[in_range] = real_gates_pos[in_range]
        gates_rpy = np.array([g.rpy for g in self.config.env.track.gates])
        real_gates_rpy = np.array([self.vicon.rpy[g] for g in gate_names])
        gates_rpy[in_range] = real_gates_rpy[in_range]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_in_range"] = in_range

        obstacles_pos = np.array([o.pos for o in self.config.env.track.obstacles])
        obstacle_names = [f"obstacle{g}" for g in range(1, len(obstacles_pos) + 1)]
        real_obstacles_pos = np.array([self.vicon.pos[o] for o in obstacle_names])
        in_range = np.linalg.norm(real_obstacles_pos - drone_pos, axis=1) < sensor_range
        obstacles_pos[in_range] = real_obstacles_pos[in_range]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_in_range"] = in_range
        return obs

    @property
    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": [], "symbolic_model": self.symbolic}

    def gate_passed(self) -> bool:
        """Check if the drone has passed the current gate.

        TODO: Implement this method.
        """
        raise NotImplementedError


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

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Take a step in the environment.

        Note:
            Sleeps for the remaining time if the step took less than the control period. This
            ensures that the environment is running at the correct frequency during deployment.
        """
        tstart = time.perf_counter()
        collective_thrust, rpy = action
        rpy_deg = np.rad2deg(rpy)
        self.cf.cmdVel(*rpy_deg, collective_thrust)
        if (dt := time.perf_counter() - tstart) < 1 / self.config.env.freq:
            rospy.sleep(1 / self.config.env.freq - dt)
        return self.obs, -1.0, False, False, self.info
