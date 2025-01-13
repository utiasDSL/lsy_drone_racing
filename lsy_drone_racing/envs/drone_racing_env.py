"""Collection of environments for drone racing simulations.

This module is a core component of the lsy_drone_racing package, providing the primary interface
between the drone racing simulation and the user's control algorithms.

It serves as a bridge between the high-level race control and the low-level drone physics
simulation. The environments defined here
(:class:`~.DroneRacingEnv` and :class:`~.DroneRacingAttitudeEnv`) expose a common interface for all
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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gymnasium
import jax
import mujoco
import numpy as np
from crazyflow import Sim
from crazyflow.sim.symbolic import symbolic_attitude
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.randomize import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)
from lsy_drone_racing.utils import check_gate_pass

if TYPE_CHECKING:
    from crazyflow.sim.structs import SimData
    from jax import Array
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region StateEnv


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

    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"

    def __init__(
        self,
        freq: int,
        sim_config: dict,
        sensor_range: float,
        track: dict | None = None,
        disturbances: dict | None = None,
        randomizations: dict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
    ):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.sim = Sim(
            physics=sim_config.physics,
            control=sim_config.get("control", "state"),
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=seed,
        )
        # Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")

        # Env settings
        self.freq = freq
        self.seed = seed
        self.symbolic = symbolic_attitude(1 / self.freq)
        self.random_resets = random_resets
        self.sensor_range = sensor_range
        self.gates, self.obstacles, self.drone = self.load_track(track)
        self.n_gates = len(track.gates)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        self.randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(13,))
        n_obstacles = len(track.obstacles)
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "target_gate": spaces.Discrete(self.n_gates, start=-1),
                "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_gates, 3)),
                "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(self.n_gates, 3)),
                "gates_visited": spaces.Box(low=0, high=1, shape=(self.n_gates,), dtype=bool),
                "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
                "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
            }
        )
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        pos_low, pos_high = np.array([-3, -3, 0]), np.array([3, 3, 2.5])
        self.bounds = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high, dtype=np.float64),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max, dtype=np.float64),
            }
        )

        # Helper variables
        self.target_gate = 0
        self._steps = 0
        self._last_drone_pos = np.zeros(3)
        self.contact_mask = np.ones((self.sim.n_worlds, 25), dtype=bool)
        self.contact_mask[..., 0] = 0  # Ignore contacts with the floor
        self.gates_visited = np.array([False] * self.n_gates)
        self.obstacles_visited = np.array([False] * n_obstacles)

        # Compile the reset and step functions with custom hooks
        self.setup_sim()

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
        if not self.random_resets:
            self.np_random = np.random.default_rng(seed=self.seed)
            self.sim.seed(self.seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
            self.sim.seed(seed)
        # Randomization of gates, obstacles and drones is compiled into the sim reset function with
        # the sim.reset_hook function, so we don't need to explicitly do it here
        self.sim.reset()
        # Reset internal variables
        self.target_gate = 0
        self._steps = 0
        self._last_drone_pos = self.sim.data.states.pos[0, 0]
        # Return info with additional reset information
        info = self.info()
        info["sim_freq"] = self.sim.data.core.freq
        info["low_level_ctrl_freq"] = self.sim.data.controls.attitude_freq
        info["drone_mass"] = self.sim.default_data.params.mass[0, 0, 0]
        info["env_freq"] = self.freq
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
        action = action.reshape((1, 1, 13))
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, (1, 1, 13))
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        self.sim.state_control(action)
        self.sim.step(self.sim.freq // self.freq)
        self.target_gate += self.gate_passed()
        if self.target_gate == self.n_gates:
            self.target_gate = -1
        self._last_drone_pos = self.sim.data.states.pos[0, 0]
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
        obs["target_gate"] = self.target_gate if self.target_gate < len(self.gates) else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        drone_pos = self.sim.data.states.pos[0, 0]
        dist = np.linalg.norm(drone_pos[:2] - self.gates["nominal_pos"][:, :2])
        in_range = dist < self.sensor_range
        self.gates_visited = np.logical_or(self.gates_visited, in_range)

        gates_pos = self.gates["nominal_pos"].copy()
        gates_pos[self.gates_visited] = self.gates["pos"][self.gates_visited]
        gates_rpy = self.gates["nominal_rpy"].copy()
        gates_rpy[self.gates_visited] = self.gates["rpy"][self.gates_visited]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_visited"] = self.gates_visited

        dist = np.linalg.norm(drone_pos[:2] - self.obstacles["nominal_pos"][:, :2])
        in_range = dist < self.sensor_range
        self.obstacles_visited = np.logical_or(self.obstacles_visited, in_range)

        obstacles_pos = self.obstacles["nominal_pos"].copy()
        obstacles_pos[self.obstacles_visited] = self.obstacles["pos"][self.obstacles_visited]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_visited"] = self.obstacles_visited
        # TODO: Decide on observation disturbances
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
        }
        if state not in self.bounds:  # Drone is out of bounds
            return True
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
        return gates, obstacles, drone

    def setup_sim(self):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        self._load_track_into_sim(self.gates, self.obstacles)
        pos = self.drone["pos"].reshape(self.sim.data.states.pos.shape)
        quat = self.drone["quat"].reshape(self.sim.data.states.quat.shape)
        vel = self.drone["vel"].reshape(self.sim.data.states.vel.shape)
        rpy_rates = self.drone["rpy_rates"].reshape(self.sim.data.states.rpy_rates.shape)
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates)
        self.sim.data = self.sim.data.replace(states=states)
        self.sim.reset_hook = build_reset_hook(
            self.randomizations, self.gates["mocap_ids"], self.obstacles["mocap_ids"]
        )
        if "dynamics" in self.disturbances:
            self.sim.disturbance_fn = build_dynamics_disturbance_fn(self.disturbances["dynamics"])
        self.sim.build(mjx=False, data=False)  # Save the reset state and rebuild the reset function

    def _load_track_into_sim(self, gates: dict, obstacles: dict):
        """Load the track into the simulation."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        spec = self.sim.spec
        frame = spec.worldbody.add_frame()
        n_gates, n_obstacles = len(gates["pos"]), len(obstacles["pos"])
        for i in range(n_gates):
            gate = frame.attach_body(gate_spec.find_body("gate"), "", f":{i}")
            gate.pos = gates["pos"][i]
            gate.quat = R.from_euler("xyz", gates["rpy"][i]).as_quat()[[3, 0, 1, 2]]  # MuJoCo order
            gate.mocap = True  # Make mocap to modify the position of static bodies during sim
        for i in range(n_obstacles):
            obstacle = frame.attach_body(obstacle_spec.find_body("obstacle"), "", f":{i}")
            obstacle.pos = obstacles["pos"][i]
            obstacle.mocap = True
        self.sim.build(data=False, default_data=False)
        # Save the ids and mocap ids of the gates and obstacles
        mj_model = self.sim.mj_model
        gates["ids"] = [mj_model.body(f"gate:{i}").id for i in range(n_gates)]
        gates["mocap_ids"] = [int(mj_model.body(f"gate:{i}").mocapid) for i in range(n_gates)]
        obstacles["ids"] = [mj_model.body(f"obstacle:{i}").id for i in range(n_obstacles)]
        obstacles["mocap_ids"] = [
            int(mj_model.body(f"obstacle:{i}").mocapid) for i in range(n_obstacles)
        ]

    def gate_passed(self) -> bool:
        """Check if the drone has passed a gate.

        Returns:
            True if the drone has passed a gate, else False.
        """
        if self.n_gates <= 0 or self.target_gate >= self.n_gates or self.target_gate == -1:
            return False
        gate_mj_id = self.gates["mocap_ids"][self.target_gate]
        gate_pos = self.sim.data.mjx_data.mocap_pos[0, gate_mj_id].squeeze()
        gate_rot = R.from_quat(self.sim.data.mjx_data.mocap_quat[0, gate_mj_id], scalar_first=True)
        drone_pos = self.sim.data.states.pos[0, 0]
        gate_size = (0.45, 0.45)
        return check_gate_pass(gate_pos, gate_rot, gate_size, drone_pos, self._last_drone_pos)

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()


# region AttitudeEnv


class DroneRacingAttitudeEnv(DroneRacingEnv):
    """Drone racing environment with a collective thrust attitude command interface.

    The action space consists of the collective thrust and body-fixed attitude commands
    [collective_thrust, roll, pitch, yaw].
    """

    def __init__(
        self,
        freq: int,
        sim_config: dict,
        sensor_range: float,
        track: dict | None = None,
        disturbances: dict | None = None,
        randomizations: dict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
    ):
        """Initialize the DroneRacingAttitudeEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        sim_config.control = "attitude"
        super().__init__(
            freq, sim_config, sensor_range, track, disturbances, randomizations, random_resets, seed
        )
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
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, (1, 1, 4))
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        self.sim.attitude_control(action.reshape((1, 1, 4)).astype(np.float32))
        self.sim.step(self.sim.freq // self.freq)
        self.target_gate += self.gate_passed()
        if self.target_gate == self.n_gates:
            self.target_gate = -1
        self._last_drone_pos = self.sim.data.states.pos[0, 0]
        return self.obs(), self.reward(), self.terminated(), False, self.info()


# region Factories


def rng_spec2fn(fn_spec: dict) -> Callable:
    """Convert a function spec to a wrapped and scaled function from jax.random."""
    offset, scale = np.array(fn_spec.get("offset", 0)), np.array(fn_spec.get("scale", 1))
    kwargs = fn_spec.get("kwargs", {})
    if "shape" in kwargs:
        raise KeyError("Shape must not be specified for randomization functions.")
    kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    jax_fn = partial(getattr(jax.random, fn_spec["fn"]), **kwargs)

    def random_fn(*args: Any, **kwargs: Any) -> Array:
        return jax_fn(*args, **kwargs) * scale + offset

    return random_fn


def build_reset_hook(
    randomizations: dict, gate_mocap_ids: list[int], obstacle_mocap_ids: list[int]
) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = []
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns.append(randomize_drone_pos_fn(rng))
            case "drone_rpy":
                randomization_fns.append(randomize_drone_quat_fn(rng))
            case "drone_mass":
                randomization_fns.append(randomize_drone_mass_fn(rng))
            case "drone_inertia":
                randomization_fns.append(randomize_drone_inertia_fn(rng))
            case "gate_pos":
                randomization_fns.append(randomize_gate_pos_fn(rng, gate_mocap_ids))
            case "gate_rpy":
                randomization_fns.append(randomize_gate_rpy_fn(rng, gate_mocap_ids))
            case "obstacle_pos":
                randomization_fns.append(randomize_obstacle_pos_fn(rng, obstacle_mocap_ids))
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_hook(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_hook


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=states.force + fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance
