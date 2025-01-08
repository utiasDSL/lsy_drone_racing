"""Collection of environments for drone racing simulations.

This module is a core component of the lsy_drone_racing package, providing the primary interface
between the drone racing simulation and the user's control algorithms.

It serves as a bridge between the high-level race control and the low-level drone physics
simulation. The environments defined here
(:class:`~.DroneRacingEnv` and :class:`~.DroneRacingThrustEnv`) expose a common interface for all
controller types, allowing for easy integration and testing of different control algorithms,
comparison of control strategies, and deployment on our hardware.

Key roles in the project:

* Abstraction Layer: Provides a standardized Gymnasium interface for interacting with the
  drone racing simulation, abstracting away the underlying physics engine.
* State Management: Handles the tracking of race progress, gate passages, and termination
  conditions.
* Observation Processing: Manages the transformation of raw simulation data into structured
  observations suitable for control algorithms.
* Action Interpretation: Translates high-level control commands into appropriate inputs for the
  underlying simulation.
* Configuration Interface: Allows for easy customization of race scenarios, environmental
  conditions, and simulation parameters.
"""

from __future__ import annotations

import copy as copy
import logging
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import check_gate_pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingEnv(gymnasium.Env):
    """A Gymnasium environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It uses the Sim class for physics simulation and supports
    various configuration options for randomization, disturbances, and physics models.

    The environment provides:
    - A customizable track with gates and obstacles
    - Configurable simulation and control frequencies
    - Support for different physics models (e.g., PyBullet, mathematical dynamics)
    - Randomization of drone properties and initial conditions
    - Disturbance modeling for realistic flight conditions
    - Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:
    - "pos": Drone position
    - "rpy": Drone orientation (roll, pitch, yaw)
    - "vel": Drone linear velocity
    - "ang_vel": Drone angular velocity
    - "gates.pos": Positions of the gates
    - "gates.rpy": Orientations of the gates
    - "gates.visited": Flags indicating if the drone already was/ is in the sensor range of the gates and the true position is known
    - "obstacles.pos": Positions of the obstacles
    - "obstacles.visited": Flags indicating if the drone already was/ is in the sensor range of the obstacles and the true position is known
    - "target_gate": The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller.
    """

    CONTROLLER = "mellinger"  # specifies controller type

    def __init__(self, config: dict):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.config = config
        self.sim = Sim(
            track=config.env.track,
            sim_freq=config.sim.sim_freq,
            ctrl_freq=config.sim.ctrl_freq,
            disturbances=getattr(config.sim, "disturbances", {}),
            randomization=getattr(config.env, "randomization", {}),
            gui=config.sim.gui,
            n_drones=1,
            physics=config.sim.physics,
        )
        self.sim.seed(config.env.seed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(13,))
        n_gates, n_obstacles = len(self.sim.gates), len(self.sim.obstacles)
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "target_gate": spaces.Discrete(n_gates, start=-1),
                "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
                "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(n_gates, 3)),
                "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=np.bool_),
                "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
                "obstacles_visited": spaces.Box(
                    low=0, high=1, shape=(n_obstacles,), dtype=np.bool_
                ),
            }
        )
        self.target_gate = 0
        self.symbolic = self.sim.symbolic() if config.env.symbolic else None
        self._steps = 0
        self._last_drone_pos = np.zeros(3)

        self.gates_visited = np.array([False] * len(config.env.track.gates))
        self.obstacles_visited = np.array([False] * len(config.env.track.obstacles))

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Observation and info.
        """
        # The system identification model is based on the attitude control interface. We cannot
        # support its use with the full state control interface
        if self.config.env.reseed:
            self.sim.seed(self.config.env.seed)
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        self.target_gate = 0
        self._steps = 0
        self.sim.drone.reset(self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel)
        self._last_drone_pos[:] = self.sim.drone.pos
        if self.sim.n_drones > 1:
            raise NotImplementedError("Firmware wrapper does not support multiple drones.")
        info = self.info
        info["sim_freq"] = self.config.sim.sim_freq
        info["low_level_ctrl_freq"] = self.config.sim.ctrl_freq
        info["drone_mass"] = self.sim.drone.nominal_params.mass
        info["env_freq"] = self.config.env.freq
        return self.obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        assert (
            self.config.sim.physics != PhysicsMode.SYS_ID
        ), "sys_id model not supported for full state control interface"
        action = action.astype(np.float64)  # Drone firmware expects float64
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.sim.drone.full_state_cmd(pos, vel, acc, yaw, rpy_rate)
        collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info

    def _inner_step_loop(self) -> bool:
        """Run the desired action for multiple simulation sub-steps.

        The outer controller is called at a lower frequency than the firmware loop. Each environment
        step therefore consists of multiple simulation steps. At each simulation step, the
        lower-level controller is called to compute the thrust and attitude commands.

        Returns:
            True if a collision occured at any point during the simulation steps, else False.
        """
        thrust = self.sim.drone.desired_thrust
        collision = False
        while (
            self.sim.drone.tick / self.sim.drone.firmware_freq
            < (self._steps + 1) / self.config.env.freq
        ):
            self.sim.step(thrust)
            self.target_gate += self.gate_passed()
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1
            collision |= bool(self.sim.collisions)
            pos, rpy, vel = self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel
            thrust = self.sim.drone.step_controller(pos, rpy, vel)[::-1]
        self.sim.drone.desired_thrust[:] = thrust
        self._last_drone_pos[:] = self.sim.drone.pos
        self._steps += 1
        return collision

    @property
    def obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the observation of the environment."""
        obs = {
            "pos": self.sim.drone.pos.astype(np.float32),
            "rpy": self.sim.drone.rpy.astype(np.float32),
            "vel": self.sim.drone.vel.astype(np.float32),
            "ang_vel": self.sim.drone.ang_vel.astype(np.float32),
        }
        obs["ang_vel"][:] = R.from_euler("xyz", obs["rpy"]).apply(obs["ang_vel"], inverse=True)

        gates = self.sim.gates
        obs["target_gate"] = self.target_gate if self.target_gate < len(gates) else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        in_range = self.sim.in_range(gates, self.sim.drone, self.config.env.sensor_range)
        self.gates_visited = np.logical_or(self.gates_visited, in_range)

        gates_pos = np.stack([g["nominal.pos"] for g in gates.values()])
        gates_pos[self.gates_visited] = np.stack([g["pos"] for g in gates.values()])[
            self.gates_visited
        ]
        gates_rpy = np.stack([g["nominal.rpy"] for g in gates.values()])
        gates_rpy[self.gates_visited] = np.stack([g["rpy"] for g in gates.values()])[
            self.gates_visited
        ]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_visited"] = self.gates_visited

        obstacles = self.sim.obstacles
        in_range = self.sim.in_range(obstacles, self.sim.drone, self.config.env.sensor_range)
        self.obstacles_visited = np.logical_or(self.obstacles_visited, in_range)

        obstacles_pos = np.stack([o["nominal.pos"] for o in obstacles.values()])
        obstacles_pos[self.obstacles_visited] = np.stack([o["pos"] for o in obstacles.values()])[
            self.obstacles_visited
        ]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_visited"] = self.obstacles_visited

        if "observation" in self.sim.disturbances:
            obs = self.sim.disturbances["observation"].apply(obs)
        return obs

    @property
    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 if self.target_gate != -1 else 0.0

    @property
    def terminated(self) -> bool:
        """Check if the episode is terminated.

        Returns:
            True if the drone is out of bounds, colliding with an obstacle, or has passed all gates,
            else False.
        """
        state = {k: getattr(self.sim.drone, k).copy() for k in self.sim.state_space}
        state["ang_vel"] = R.from_euler("xyz", state["rpy"]).apply(state["ang_vel"], inverse=True)
        if state not in self.sim.state_space:
            return True  # Drone is out of bounds
        if self.sim.collisions:
            return True
        if self.target_gate == -1:  # Drone has passed all gates
            return True
        return False

    @property
    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": self.sim.collisions, "symbolic_model": self.symbolic}

    def gate_passed(self) -> bool:
        """Check if the drone has passed a gate.

        Returns:
            True if the drone has passed a gate, else False.
        """
        if self.sim.n_gates > 0 and self.target_gate < self.sim.n_gates and self.target_gate != -1:
            gate_pos = self.sim.gates[self.target_gate]["pos"]
            gate_rot = R.from_euler("xyz", self.sim.gates[self.target_gate]["rpy"])
            drone_pos = self.sim.drone.pos
            last_drone_pos = self._last_drone_pos
            gate_size = (0.45, 0.45)
            return check_gate_pass(gate_pos, gate_rot, gate_size, drone_pos, last_drone_pos)
        return False

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()


class DroneRacingThrustEnv(DroneRacingEnv):
    """Drone racing environment with a collective thrust attitude command interface.

    The action space consists of the collective thrust and body-fixed attitude commands
    [collective_thrust, roll, pitch, yaw].
    """

    def __init__(self, config: dict):
        """Initialize the DroneRacingThrustEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__(config)
        bounds = np.array([1, np.pi, np.pi, np.pi], dtype=np.float32)
        self.action_space = spaces.Box(low=-bounds, high=bounds)

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Step the drone racing environment with a thrust and attitude command.

        Args:
            action: Thrust command [thrust, roll, pitch, yaw].
        """
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        action = action.astype(np.float64)
        collision = False
        # We currently need to differentiate between the sys_id backend and all others because the
        # simulation step size is different for the sys_id backend (we do not substep in the
        # identified model). In future iterations, the sim API should be flexible to handle both
        # cases without an explicit step_sys_id function.
        if self.config.sim.physics == "sys_id":
            cmd_thrust, cmd_rpy = action[0], action[1:]
            self.sim.step_sys_id(cmd_thrust, cmd_rpy, 1 / self.config.env.freq)
            self.target_gate += self.gate_passed()
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1
            self._last_drone_pos[:] = self.sim.drone.pos
        else:
            # Crazyflie firmware expects negated pitch command. TODO: Check why this is the case and
            # fix this on the firmware side if possible.
            cmd_thrust, cmd_rpy = action[0], action[1:] * np.array([1, -1, 1])
            self.sim.drone.collective_thrust_cmd(cmd_thrust, cmd_rpy)
            collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info
