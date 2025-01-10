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
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import gymnasium
import mujoco
import numpy as np
from crazyflow import Sim
from crazyflow.sim.sim import identity
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.utils import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
)
from lsy_drone_racing.sim.noise import NoiseList
from lsy_drone_racing.utils import check_gate_pass

if TYPE_CHECKING:
    from crazyflow.sim.structs import SimData
    from jax import Array
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
    - "gates.visited": Flags indicating if the drone already was/ is in the sensor range of the
    gates and the true position is known
    - "obstacles.pos": Positions of the obstacles
    - "obstacles.visited": Flags indicating if the drone already was/ is in the sensor range of the
      obstacles and the true position is known
    - "target_gate": The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller.
    """

    gate_spec_path = Path(__file__).parents[1] / "sim/assets/gate.urdf"
    obstacle_spec_path = Path(__file__).parents[1] / "sim/assets/obstacle.urdf"

    def __init__(self, config: dict):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.config = config
        self.sim = Sim(
            n_worlds=1,
            n_drones=1,
            physics=config.sim.physics,
            control=config.sim.get("control", "state"),
            freq=config.sim.sim_freq,
            state_freq=config.env.freq,
            attitude_freq=config.sim.attitude_freq,
            rng_key=config.env.seed,
        )
        if config.sim.sim_freq % config.env.freq != 0:
            raise ValueError(f"({config.sim.sim_freq=}) is no multiple of ({config.env.freq=})")

        self.action_space = spaces.Box(low=-1, high=1, shape=(13,))
        n_gates, n_obstacles = len(config.env.track.gates), len(config.env.track.obstacles)
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
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        pos_low, pos_high = np.array([-3, -3, 0]), np.array([3, 3, 2.5])
        self.state_space = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high, dtype=np.float64),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max, dtype=np.float64),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            }
        )

        self.target_gate = 0
        self.symbolic = self.sim.symbolic() if config.env.symbolic else None
        self._steps = 0
        self._last_drone_pos = np.zeros(3)
        self.gates, self.obstacles, self.drone = self.load_track(config.env.track)
        self.n_gates = len(config.env.track.gates)
        self.disturbances = self.load_disturbances(config.env.get("disturbances", None))
        self.randomization = self.load_randomizations(config.env.get("randomization", None))
        self.contact_mask = np.ones((self.sim.n_worlds, 29), dtype=bool)
        self.contact_mask[..., 0] = 0  # Ignore contacts with the floor

        self.setup_sim()

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
        if self.config.env.reseed:
            self.sim.seed(self.config.env.seed)
        if seed is not None:
            self.sim.seed(seed)
        # Randomization of gates, obstacles and drones is compiled into the sim reset function with
        # the sim.reset_hook function, so we don't need to explicitly do it here
        self.sim.reset()
        # TODO: Add randomization of gates, obstacles, drone, and disturbances
        self.target_gate = 0
        self._steps = 0
        self._last_drone_pos[:] = self.sim.data.states.pos[0, 0]
        info = self.info()
        info["sim_freq"] = self.sim.data.core.freq
        info["low_level_ctrl_freq"] = self.sim.data.controls.attitude_freq
        info["drone_mass"] = self.sim.default_data.params.mass[0, 0, 0]
        info["env_freq"] = self.config.env.freq
        return self.obs(), info

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
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        # TODO: Add action noise
        # TODO: Check why sim is being compiled twice
        self.sim.state_control(action.reshape((1, 1, 13)).astype(np.float32))
        self.sim.step(self.sim.freq // self.config.env.freq)
        self.target_gate += self.gate_passed()
        if self.target_gate == self.n_gates:
            self.target_gate = -1
        self._last_drone_pos[:] = self.sim.data.states.pos[0, 0]
        return self.obs(), self.reward(), self.terminated(), False, self.info()

    def render(self):
        """Render the environment."""
        self.sim.render()

    def obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the observation of the environment."""
        obs = {
            "pos": np.array(self.sim.data.states.pos[0, 0], dtype=np.float32),
            "rpy": R.from_quat(self.sim.data.states.quat[0, 0]).as_euler("xyz").astype(np.float32),
            "vel": np.array(self.sim.data.states.vel[0, 0], dtype=np.float32),
            "ang_vel": np.array(self.sim.data.states.rpy_rates[0, 0], dtype=np.float32),
        }
        obs["ang_vel"][:] = R.from_euler("xyz", obs["rpy"]).apply(obs["ang_vel"], inverse=True)

        obs["target_gate"] = self.target_gate if self.target_gate < len(self.gates) else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        drone_pos = self.sim.data.states.pos[0, 0]
        dist = np.linalg.norm(drone_pos[:2] - self.gates["nominal_pos"][:, :2])
        in_range = dist < self.config.env.sensor_range
        self.gates_visited = np.logical_or(self.gates_visited, in_range)

        gates_pos = self.gates["nominal_pos"].copy()
        gates_pos[self.gates_visited] = self.gates["pos"][self.gates_visited]
        gates_rpy = self.gates["nominal_rpy"].copy()
        gates_rpy[self.gates_visited] = self.gates["rpy"][self.gates_visited]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_visited"] = self.gates_visited

        dist = np.linalg.norm(drone_pos[:2] - self.obstacles["nominal_pos"][:, :2])
        in_range = dist < self.config.env.sensor_range
        self.obstacles_visited = np.logical_or(self.obstacles_visited, in_range)

        obstacles_pos = self.obstacles["nominal_pos"].copy()
        obstacles_pos[self.obstacles_visited] = self.obstacles["pos"][self.obstacles_visited]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_visited"] = self.obstacles_visited

        if "observation" in self.disturbances:
            obs = self.disturbances["observation"].apply(obs)
        return obs

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

    def terminated(self) -> bool:
        """Check if the episode is terminated.

        Returns:
            True if the drone is out of bounds, colliding with an obstacle, or has passed all gates,
            else False.
        """
        quat = self.sim.data.states.quat[0, 0]
        state = {
            "pos": np.array(self.sim.data.states.pos[0, 0], dtype=np.float32),
            "rpy": np.array(R.from_quat(quat).as_euler("xyz"), dtype=np.float32),
            "vel": np.array(self.sim.data.states.vel[0, 0], dtype=np.float32),
            "ang_vel": np.array(
                R.from_quat(quat).apply(self.sim.data.states.rpy_rates[0, 0], inverse=True),
                dtype=np.float32,
            ),
        }
        if state not in self.state_space:
            return True  # Drone is out of bounds
        if np.logical_and(self.sim.contacts("drone:0"), self.contact_mask).any():
            return True
        if self.sim.data.states.pos[0, 0, 2] < 0.0:
            return True
        if self.target_gate == -1:  # Drone has passed all gates
            return True
        return False

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": self.sim.contacts("drone:0").any(), "symbolic_model": self.symbolic}

    def load_track(self, track: dict) -> tuple[dict, dict, dict]:
        """Load the track from the config file."""
        gate_pos = np.array([g["pos"] for g in track.gates])
        gate_rpy = np.array([g["rpy"] for g in track.gates])
        gates = {"pos": gate_pos, "rpy": gate_rpy, "nominal_pos": gate_pos, "nominal_rpy": gate_rpy}
        obstacle_pos = np.array([o["pos"] for o in track.obstacles])
        obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos}
        drone = {
            k: np.array([track.drone.get(k)], dtype=np.float32)
            for k in ("pos", "rpy", "vel", "rpy_rates")
        }
        drone["quat"] = R.from_euler("xyz", drone["rpy"]).as_quat()
        # Load the models into the simulation and set their positions
        self._load_track_into_sim(gates, obstacles)
        return gates, obstacles, drone

    def _load_track_into_sim(self, gates: dict, obstacles: dict):
        """Load the track into the simulation."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        spec = self.sim.spec
        frame = spec.worldbody.add_frame()
        for i in range(len(gates["pos"])):
            gate = frame.attach_body(gate_spec.find_body("world"), "", f":g{i}")
            gate.pos = gates["pos"][i]
            quat = R.from_euler("xyz", gates["rpy"][i]).as_quat()
            gate.quat = quat[[3, 0, 1, 2]]  # MuJoCo uses wxyz order instead of xyzw
        for i in range(len(obstacles["pos"])):
            obstacle = frame.attach_body(obstacle_spec.find_body("world"), "", f":o{i}")
            obstacle.pos = obstacles["pos"][i]
        self.sim.build()

    def load_disturbances(self, disturbances: dict | None = None) -> dict:
        """Load the disturbances from the config."""
        dist = {}  # TODO: Add jax disturbances for the simulator dynamics
        if disturbances is None:  # Default: no passive disturbances.
            return dist
        for mode, spec in disturbances.items():
            dist[mode] = NoiseList.from_specs([spec])
        return dist

    def load_randomizations(self, randomizations: dict | None = None) -> dict:
        """Load the randomization from the config."""
        if randomizations is None:
            return {}
        return {}

    def setup_sim(self):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        pos = self.drone["pos"].reshape(self.sim.data.states.pos.shape)
        quat = self.drone["quat"].reshape(self.sim.data.states.quat.shape)
        vel = self.drone["vel"].reshape(self.sim.data.states.vel.shape)
        rpy_rates = self.drone["rpy_rates"].reshape(self.sim.data.states.rpy_rates.shape)
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates)
        self.sim.data = self.sim.data.replace(states=states)
        reset_hook = build_reset_hook(self.randomization)
        self.sim.reset_hook = reset_hook
        self.sim.build(mjx=False, data=False)  # Save the reset state and rebuild the reset function

    def gate_passed(self) -> bool:
        """Check if the drone has passed a gate.

        Returns:
            True if the drone has passed a gate, else False.
        """
        if self.n_gates > 0 and self.target_gate < self.n_gates and self.target_gate != -1:
            gate_pos = self.gates["pos"][self.target_gate]
            gate_rot = R.from_euler("xyz", self.gates["rpy"][self.target_gate])
            drone_pos = self.sim.data.states.pos[0, 0]
            last_drone_pos = self._last_drone_pos
            gate_size = (0.45, 0.45)
            return check_gate_pass(gate_pos, gate_rot, gate_size, drone_pos, last_drone_pos)
        return False

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()


def build_reset_hook(randomizations: dict) -> Callable[[SimData, Array[bool]], SimData]:
    """Build the reset hook for the simulation."""
    modify_drone_pos = identity
    if "drone_pos" in randomizations:
        modify_drone_pos = randomize_drone_pos_fn(randomizations["drone_pos"])
    modify_drone_quat = identity
    if "drone_rpy" in randomizations:
        modify_drone_quat = randomize_drone_quat_fn(randomizations["drone_rpy"])
    modify_drone_mass = identity
    if "drone_mass" in randomizations:
        modify_drone_mass = randomize_drone_mass_fn(randomizations["drone_mass"])
    modify_drone_inertia = identity
    if "drone_inertia" in randomizations:
        modify_drone_inertia = randomize_drone_inertia_fn(randomizations["drone_inertia"])
    modify_gate_pos = identity
    if "gate_pos" in randomizations:
        modify_gate_pos = randomize_gate_pos_fn(randomizations["gate_pos"])
    modify_gate_rpy = identity
    if "gate_rpy" in randomizations:
        modify_gate_rpy = randomize_gate_rpy_fn(randomizations["gate_rpy"])
    modify_obstacle_pos = identity
    if "obstacle_pos" in randomizations:
        modify_obstacle_pos = randomize_obstacle_pos_fn(randomizations["obstacle_pos"])

    def reset_hook(data: SimData, mask: Array[bool]) -> SimData:
        data = modify_drone_pos(data, mask)
        data = modify_drone_quat(data, mask)
        data = modify_drone_mass(data, mask)
        data = modify_drone_inertia(data, mask)
        data = modify_gate_pos(data, mask)
        data = modify_gate_rpy(data, mask)
        data = modify_obstacle_pos(data, mask)
        return data

    return reset_hook


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
        config.sim.control = "attitude"
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
        # TODO: Add action noise
        # TODO: Check why sim is being compiled twice
        self.sim.attitude_control(action.reshape((1, 1, 4)).astype(np.float32))
        self.sim.step(self.sim.freq // self.config.env.freq)
        self.target_gate += self.gate_passed()
        if self.target_gate == self.n_gates:
            self.target_gate = -1
        self._last_drone_pos[:] = self.sim.data.states.pos[0, 0]
        return self.obs(), self.reward(), self.terminated(), False, self.info()
