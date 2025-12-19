"""Core environment for drone racing simulations.

This module provides the shared logic for simulating drone racing environments. It defines a core
environment class that wraps our drone simulation, drone control, gate tracking, and collision
detection. The module serves as the base for both single-drone and multi-drone racing environments.

The environment is designed to be configurable, supporting:

* Different control modes (state or attitude)
* Customizable tracks with gates and obstacles
* Various randomization options for robust policy training
* Disturbance modeling for realistic flight conditions
* Vectorized execution for parallel training

This module is primarily used as a base for the higher-level environments in
:mod:`~lsy_drone_racing.envs.drone_race` and :mod:`~lsy_drone_racing.envs.multi_drone_race`,
which provide Gymnasium-compatible interfaces for reinforcement learning, MPC and other control
techniques.
"""

from __future__ import annotations

import copy as copy
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from drone_controllers.mellinger.params import ForceTorqueParams
from flax.struct import dataclass
from gymnasium import spaces

from lsy_drone_racing.envs.randomize import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)
from lsy_drone_racing.envs.utils import gate_passed, generate_random_track, load_track

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array, Device
    from ml_collections import ConfigDict
    from mujoco import MjSpec
    from mujoco.mjx import Data
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region EnvData


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment.

    This dataclass stores the dynamic and static state of the environment that is not directly
    part of the physics simulation. It includes information about gate progress, drone status,
    and environment boundaries. Static variables are initialized once and do not change during the
    episode.

    Args:
        target_gate: Current target gate index for each drone in each environment
        gates_visited: Boolean flags indicating which gates have been visited by each drone
        obstacles_visited: Boolean flags indicating which obstacles have been detected
        last_drone_pos: Previous positions of drones, used for gate passing detection
        marked_for_reset: Flags indicating which environments need to be reset
        disabled_drones: Flags indicating which drones have crashed or are otherwise disabled
        contact_masks: Masks for contact detection between drones and objects
        pos_limit_low: Lower position limits for the environment
        pos_limit_high: Upper position limits for the environment
        gate_mj_ids: MuJoCo IDs for the gates
        obstacle_mj_ids: MuJoCo IDs for the obstacles
        max_episode_steps: Maximum number of steps per episode
        sensor_range: Range at which drones can detect gates and obstacles
    """

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    marked_for_reset: Array
    disabled_drones: Array
    steps: Array
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    gate_mj_ids: Array
    obstacle_mj_ids: Array
    max_episode_steps: Array
    sensor_range: Array

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_drones: int,
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        gate_mj_ids: Array,
        obstacle_mj_ids: Array,
        max_episode_steps: int,
        sensor_range: float,
        pos_limit_low: Array,
        pos_limit_high: Array,
        device: Device,
    ) -> EnvData:
        """Create a new environment data struct with default values."""
        return cls(
            target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jp.zeros((n_envs, n_drones, n_gates), dtype=bool, device=device),
            obstacles_visited=jp.zeros((n_envs, n_drones, n_obstacles), dtype=bool, device=device),
            last_drone_pos=jp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            marked_for_reset=jp.zeros(n_envs, dtype=bool, device=device),
            disabled_drones=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jp.array(contact_masks, dtype=bool, device=device),
            steps=jp.zeros(n_envs, dtype=int, device=device),
            pos_limit_low=jp.array(pos_limit_low, dtype=np.float32, device=device),
            pos_limit_high=jp.array(pos_limit_high, dtype=np.float32, device=device),
            gate_mj_ids=jp.array(gate_mj_ids, dtype=int, device=device),
            obstacle_mj_ids=jp.array(obstacle_mj_ids, dtype=int, device=device),
            max_episode_steps=jp.array([max_episode_steps], dtype=int, device=device),
            sensor_range=jp.array([sensor_range], dtype=jp.float32, device=device),
        )


def build_action_space(control_mode: Literal["state", "attitude"], drone_model: str) -> spaces.Box:
    """Create the action space for the environment.

    Args:
        control_mode: The control mode to use. Either "state" for full-state control
            or "attitude" for attitude control.
        drone_model: Drone model of the environment.

    Returns:
        A Box space representing the action space for the specified control mode.
    """
    if control_mode == "state":
        return spaces.Box(low=-1, high=1, shape=(13,))
    elif control_mode == "attitude":
        params = ForceTorqueParams.load(drone_model)
        thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
        return spaces.Box(
            np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
            np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
        )
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")


def build_observation_space(n_gates: int, n_obstacles: int) -> spaces.Dict:
    """Create the observation space for the environment.

    The observation space is a dictionary containing the drone state, gate information,
    and obstacle information.

    Args:
        n_gates: Number of gates in the environment.
        n_obstacles: Number of obstacles in the environment.
    """
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "quat": spaces.Box(low=-1, high=1, shape=(4,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "target_gate": spaces.Discrete(n_gates, start=-1),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_quat": spaces.Box(low=-1, high=1, shape=(n_gates, 4)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
    }
    return spaces.Dict(obs_spec)


# region Core Env


class RaceCoreEnv:
    """The core environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.

    The environment provides:

    * A customizable track with gates and obstacles
    * Configurable simulation and control frequencies
    * Support for different physics models (e.g., identified dynamics, analytical dynamics)
    * Randomization of drone properties and initial conditions
    * Disturbance modeling for realistic flight conditions
    * Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:

    * pos: Drone position
    * quat: Drone orientation as a quaternion (x, y, z, w)
    * vel: Drone linear velocity
    * ang_vel: Drone angular velocity
    * gates_pos: Positions of the gates
    * gates_quat: Orientations of the gates
    * gates_visited: Flags indicating if the drone already was/ is in the sensor range of the
      gates and the true position is known
    * obstacles_pos: Positions of the obstacles
    * obstacles_visited: Flags indicating if the drone already was/ is in the sensor range of the
      obstacles and the true position is known
    * target_gate: The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller, or a desired collective thrust and attitude command [collective thrust,
    roll, pitch, yaw].
    """

    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"

    def __init__(
        self,
        n_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: str | int = "random",
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the DroneRacingEnv.

        Args:
            n_envs: Number of worlds in the vectorized environment.
            n_drones: Number of drones.
            freq: Environment step frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: "random" for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode. Needs to be tracked manually for
                vectorized environments.
            device: Device used for the environment and the simulation.
        """
        super().__init__()
        if type(seed) is str:
            seed: int = np.random.SeedSequence().entropy if seed == "random" else hash(seed)
            seed &= 0xFFFFFFFF  # Limit seed to 32 bit for jax.random
        self.sim = Sim(
            n_worlds=n_envs,
            n_drones=n_drones,
            physics=sim_config.physics,
            drone_model=sim_config.drone_model,
            control=control_mode,
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=seed,
            device=device,
        )
        use_box_collision(self.sim, True)
        self.cam_config = {
            "distance": sim_config.camera_view[0],
            "azimuth": sim_config.camera_view[1],
            "elevation": sim_config.camera_view[2],
            "lookat": sim_config.camera_view[3:],
        }

        # Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")

        # Env settings
        self.freq = freq
        self.seed = seed
        self.autoreset = True  # Can be overridden by subclasses
        self.device = jax.devices(device)[0]
        self.sensor_range = sensor_range
        self.track = track
        self.gates, self.obstacles, self.drone = load_track(track)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Load the track into the simulation and compile the reset and step functions with hooks
        self._setup_sim(randomizations)

        # Create the environment data struct.
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        contact_masks = self._load_contact_masks(self.sim)
        m = self.sim.mj_model
        gate_ids = [int(m.body(f"gate:{i}").mocapid.squeeze()) for i in range(n_gates)]
        obstacle_ids = [int(m.body(f"obstacle:{i}").mocapid.squeeze()) for i in range(n_obstacles)]
        self.data = EnvData.create(
            n_envs=n_envs,
            n_drones=n_drones,
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            contact_masks=contact_masks,
            gate_mj_ids=gate_ids,
            obstacle_mj_ids=obstacle_ids,
            max_episode_steps=max_episode_steps,
            sensor_range=sensor_range,
            pos_limit_low=[-3, -3, -1e-3],
            pos_limit_high=[3, 3, 2.5],
            device=self.device,
        )
        self.randomize_track = build_track_randomization_fn(randomizations, gate_ids, obstacle_ids)

    def _reset(
        self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.
            mask: Mask of worlds to reset.

        Returns:
            Observation and info.
        """
        if seed is not None:
            self.sim.seed(seed)
            self._np_random = np.random.default_rng(seed)  # Also update gymnasium's rng
        # Randomization of the drone is compiled into the sim reset pipeline, so we don't need to
        # explicitly do it here
        self.sim.reset(mask=mask)
        key, subkey, subkey2 = jax.random.split(self.sim.data.core.rng_key, 3)
        # Generate random track
        track = generate_random_track(self.track, subkey2) if self.track.randomize else self.track
        self.gates, self.obstacles, self.drone = load_track(track)
        # Randomize the track
        self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))

        @jax.jit
        def update_sim_data(
            data: SimData, mjx_data: Data, key: jax.random.PRNGKey
        ) -> tuple[SimData, Data]:
            # Randomized drone pos
            pos = data.states.pos.at[...].set(self.drone["pos"])
            data = data.replace(states=data.states.replace(pos=pos))

            mjx_data = self.randomize_track(
                mjx_data,
                mask,
                self.gates["nominal_pos"],
                self.gates["nominal_quat"],
                self.obstacles["nominal_pos"],
                key,
            )
            return data, mjx_data

        self.sim.data, self.sim.mjx_data = update_sim_data(self.sim.data, self.sim.mjx_data, subkey)

        # Reset the environment data
        self.data = self._reset_env_data(
            self.data, self.sim.data.states.pos, self.sim.mjx_data.mocap_pos, mask
        )

        return self.obs(), self.info()

    def _step(self, action: Array) -> tuple[dict[str, Array], float, bool, bool, dict]:
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        self.apply_action(action)
        self.sim.step(self.sim.freq // self.freq)
        # Warp drones that have crashed outside the track to prevent them from interfering with
        # other drones still in the race
        self.sim.data = self._warp_disabled_drones(self.sim.data, self.data.disabled_drones)
        # Apply the environment logic. Check which drones are now disabled, check which gates have
        # been passed, and update the target gate.
        drone_pos = self.sim.data.states.pos
        mocap_pos, mocap_quat = self.sim.mjx_data.mocap_pos, self.sim.mjx_data.mocap_quat
        contacts = self.sim.contacts()
        # Get marked_for_reset before it is updated, because the autoreset needs to be based on the
        # previous flags, not the ones from the current step
        marked_for_reset = self.data.marked_for_reset
        # Apply the environment logic with updated simulation data.
        self.data = self._step_env(
            self.data, drone_pos, mocap_pos, mocap_quat, contacts, self.sim.freq
        )
        # Auto-reset envs. Add configuration option to disable for single-world envs
        if self.autoreset and marked_for_reset.any():
            self._reset(mask=marked_for_reset)
        return self.obs(), self.reward(), self.terminated(), self.truncated(), self.info()

    def apply_action(self, action: Array):
        """Apply the commanded state action to the simulation."""
        # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
        # https://github.com/jax-ml/jax/discussions/6055
        # Tracking issue:
        # https://github.com/jax-ml/jax/issues/29810
        # Forcing a copy here is less efficient, but avoids the warning.
        action = np.reshape(action, (self.sim.n_worlds, self.sim.n_drones, -1), copy=True)
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        match self.sim.control:
            case "attitude":
                self.sim.attitude_control(action)
            case "state":
                self.sim.state_control(action)
            case _:
                raise ValueError(f"Unsupported control mode: {self.sim.control}")

    def render(self):
        """Render the environment."""
        self.sim.render(cam_config=self.cam_config)

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()

    def obs(self) -> dict[str, Array]:
        """Return the observation of the environment."""
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        gates_pos, gates_quat, obstacles_pos = self._obs(
            self.sim.mjx_data.mocap_pos,
            self.sim.mjx_data.mocap_quat,
            self.data.gates_visited,
            self.data.gate_mj_ids,
            self.gates["nominal_pos"],
            self.gates["nominal_quat"],
            self.data.obstacles_visited,
            self.data.obstacle_mj_ids,
            self.obstacles["nominal_pos"],
        )
        obs = {
            "pos": self.sim.data.states.pos,
            "quat": self.sim.data.states.quat,
            "vel": self.sim.data.states.vel,
            "ang_vel": self.sim.data.states.ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> Array:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 * (self.data.target_gate == -1)  # Implicit float conversion

    def terminated(self) -> Array:
        """Check if the episode is terminated.

        Returns:
            True if all drones have been disabled, else False.
        """
        return self.data.disabled_drones

    def truncated(self) -> Array:
        """Array of booleans indicating if the episode is truncated."""
        return self._truncated(self.data.steps, self.data.max_episode_steps, self.sim.n_drones)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @staticmethod
    @jax.jit
    def _reset_env_data(
        data: EnvData, drone_pos: Array, mocap_pos: Array, mask: Array | None = None
    ) -> EnvData:
        """Reset auxiliary variables of the environment data."""
        mask = jp.ones(data.steps.shape, dtype=bool) if mask is None else mask
        target_gate = jp.where(mask[..., None], 0, data.target_gate)
        last_drone_pos = jp.where(mask[..., None, None], drone_pos, data.last_drone_pos)
        disabled_drones = jp.where(mask[..., None], False, data.disabled_drones)
        steps = jp.where(mask, 0, data.steps)
        # Check which gates are in range of the drone
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
        gates_visited = jp.where(mask[..., None, None], gates_visited, data.gates_visited)
        # And which obstacles are in range
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
        obstacles_visited = jp.where(
            mask[..., None, None], obstacles_visited, data.obstacles_visited
        )
        return data.replace(
            target_gate=target_gate,
            last_drone_pos=last_drone_pos,
            disabled_drones=disabled_drones,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            steps=steps,
            marked_for_reset=jp.where(mask, 0, data.marked_for_reset),  # Unmark after env reset
        )

    @staticmethod
    @jax.jit
    def _step_env(
        data: EnvData,
        drone_pos: Array,
        mocap_pos: Array,
        mocap_quat: Array,
        contacts: Array,
        freq: int,
    ) -> EnvData:
        """Step the environment data."""
        n_gates = len(data.gate_mj_ids)
        taken_off_drones = (data.steps > freq // 5)[:, None]  # Only activate check after 0.2s
        disabled_drones = taken_off_drones & RaceCoreEnv._disabled_drones(drone_pos, contacts, data)
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        # We need to convert the mocap quat from MuJoCo order to scipy order
        gates_quat = mocap_quat[:, data.gate_mj_ids][..., [1, 2, 3, 0]]
        # Extract the gate poses of the current target gates and check if the drones have passed
        # them between the last and current position
        gate_ids = data.gate_mj_ids[data.target_gate % n_gates]
        gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], gate_ids]
        gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], gate_ids]
        passed = gate_passed(drone_pos, data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
        # Update the target gate index. Increment by one if drones have passed a gate
        target_gate = data.target_gate + passed * ~disabled_drones
        target_gate = jp.where(target_gate >= n_gates, -1, target_gate)
        steps = data.steps + 1
        truncated = steps >= data.max_episode_steps
        marked_for_reset = jp.all(disabled_drones | truncated[..., None], axis=-1)
        # Update which gates and obstacles are or have been in range of the drone
        sensor_range = data.sensor_range
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = data.gates_visited | (jp.linalg.norm(dpos, axis=-1) < sensor_range)
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = data.obstacles_visited | (jp.linalg.norm(dpos, axis=-1) < sensor_range)
        data = data.replace(
            last_drone_pos=drone_pos,
            target_gate=target_gate,
            disabled_drones=disabled_drones,
            marked_for_reset=marked_for_reset,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            steps=steps,
        )
        return data

    @staticmethod
    @jax.jit
    def _obs(
        mocap_pos: Array,
        mocap_quat: Array,
        gates_visited: Array,
        gate_mocap_ids: Array,
        nominal_gate_pos: NDArray,
        nominal_gate_quat: NDArray,
        obstacles_visited: Array,
        obstacle_mocap_ids: Array,
        nominal_obstacle_pos: NDArray,
    ) -> tuple[Array, Array]:
        """Get the nominal or real gate positions and orientations depending on the sensor range."""
        mask, real_pos = gates_visited[..., None], mocap_pos[:, gate_mocap_ids]
        real_quat = mocap_quat[:, gate_mocap_ids][..., [1, 2, 3, 0]]
        gates_pos = jp.where(mask, real_pos[:, None], nominal_gate_pos[None, None])
        gates_quat = jp.where(mask, real_quat[:, None], nominal_gate_quat[None, None])
        mask, real_pos = obstacles_visited[..., None], mocap_pos[:, obstacle_mocap_ids]
        obstacles_pos = jp.where(mask, real_pos[:, None], nominal_obstacle_pos[None, None])
        return gates_pos, gates_quat, obstacles_pos

    @staticmethod
    @partial(jax.jit, static_argnames="n_drones")
    def _truncated(steps: Array, max_episode_steps: Array, n_drones: int) -> Array:
        return jp.tile((steps >= max_episode_steps)[..., None], (1, n_drones))

    @staticmethod
    def _disabled_drones(pos: Array, contacts: Array, data: EnvData) -> Array:
        disabled = data.disabled_drones | jp.any(pos < data.pos_limit_low, axis=-1)
        disabled = disabled | jp.any(pos > data.pos_limit_high, axis=-1)
        disabled = disabled | (data.target_gate == -1)
        contacts = jp.any(contacts[:, None, :] & data.contact_masks, axis=-1)
        disabled = disabled | contacts
        return disabled

    @staticmethod
    @jax.jit
    def _warp_disabled_drones(data: SimData, mask: Array) -> SimData:
        """Warp the disabled drones below the ground."""
        pos = jax.numpy.where(mask[..., None], -1, data.states.pos)
        return data.replace(states=data.states.replace(pos=pos))

    def _setup_sim(self, randomizations: dict):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        self._load_track_into_sim(gate_spec, obstacle_spec)
        # Set the initial drone states
        pos = self.sim.data.states.pos.at[...].set(self.drone["pos"])
        quat = self.sim.data.states.quat.at[...].set(self.drone["quat"])
        vel = self.sim.data.states.vel.at[...].set(self.drone["vel"])
        ang_vel = self.sim.data.states.ang_vel.at[...].set(self.drone["ang_vel"])
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        self.sim.data = self.sim.data.replace(states=states)
        self.sim.build_default_data()
        # Build the reset randomizations and disturbances into the sim itself
        self.sim.reset_pipeline = self.sim.reset_pipeline + (build_reset_fn(randomizations),)
        self.sim.build_reset_fn()
        if "dynamics" in self.disturbances:
            disturbance_fn = build_dynamics_disturbance_fn(self.disturbances["dynamics"])
            self.sim.step_pipeline = (
                self.sim.step_pipeline[:2] + (disturbance_fn,) + self.sim.step_pipeline[2:]
            )
            self.sim.build_step_fn()

    def _load_track_into_sim(self, gate_spec: MjSpec, obstacle_spec: MjSpec):
        """Load the track into the simulation."""
        frame = self.sim.spec.worldbody.add_frame()
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        for i in range(n_gates):
            gate_body = gate_spec.body("gate")
            if gate_body is None:
                raise ValueError("Gate body not found in gate spec")
            gate = frame.attach_body(gate_body, "", f":{i}")
            gate.pos = self.gates["pos"][i]
            # Convert from scipy order to MuJoCo order
            gate.quat = self.gates["quat"][i][[3, 0, 1, 2]]
            gate.mocap = True  # Make mocap to modify the position of static bodies during sim
        for i in range(n_obstacles):
            obstacle_body = obstacle_spec.body("obstacle")
            if obstacle_body is None:
                raise ValueError("Obstacle body not found in obstacle spec")
            obstacle = frame.attach_body(obstacle_body, "", f":{i}")
            obstacle.pos = self.obstacles["pos"][i]
            obstacle.mocap = True
        self.sim.build_mjx()

    @staticmethod
    def _load_contact_masks(sim: Sim) -> Array:  # , data: EnvData
        """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
        sim.contacts()  # Trigger initial contact information computation
        contact = sim.mjx_data._impl.contact
        n_contacts = len(contact.geom1[0])
        masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)
        # We only need one world to create the mask
        geom1, geom2 = (contact.geom1[0], contact.geom2[0])
        for i in range(sim.n_drones):
            geom_start = sim.mj_model.body_geomadr[sim.mj_model.body(f"drone:{i}").id]
            geom_count = sim.mj_model.body_geomnum[sim.mj_model.body(f"drone:{i}").id]
            geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
            geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
            masks[i, :] = geom1_valid | geom2_valid
        geom_start = sim.mj_model.body_geomadr[sim.mj_model.body("world").id]
        geom_count = sim.mj_model.body_geomnum[sim.mj_model.body("world").id]
        geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
        geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)

        masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
        return masks


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


def build_reset_fn(randomizations: dict) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns += (randomize_drone_pos_fn(rng),)
            case "drone_rpy":
                randomization_fns += (randomize_drone_quat_fn(rng),)
            case "drone_mass":
                randomization_fns += (randomize_drone_mass_fn(rng),)
            case "drone_inertia":
                randomization_fns += (randomize_drone_inertia_fn(rng),)
            case "gate_pos" | "gate_rpy" | "obstacle_pos":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_fn(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_fn


def build_track_randomization_fn(
    randomizations: dict, gate_mocap_ids: list[int], obstacle_mocap_ids: list[int]
) -> Callable[[Data, Array, jax.random.PRNGKey], Data]:
    """Build the track randomization function for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "gate_pos":
                randomization_fns += (randomize_gate_pos_fn(rng, gate_mocap_ids),)
            case "gate_rpy":
                randomization_fns += (randomize_gate_rpy_fn(rng, gate_mocap_ids),)
            case "obstacle_pos":
                randomization_fns += (randomize_obstacle_pos_fn(rng, obstacle_mocap_ids),)
            case "drone_pos" | "drone_rpy" | "drone_mass" | "drone_inertia":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    @jax.jit
    def track_randomization(
        data: Data,
        mask: Array,
        nominal_gate_pos: Array,
        nominal_gate_quat: Array,
        nominal_obstacle_pos: Array,
        key: jax.random.PRNGKey,
    ) -> Data:
        gate_quat = jp.roll(nominal_gate_quat, 1, axis=-1)  # Convert from scipy to MuJoCo order

        # Reset to default track positions first
        data = data.replace(mocap_pos=data.mocap_pos.at[:, gate_mocap_ids].set(nominal_gate_pos))
        data = data.replace(mocap_quat=data.mocap_quat.at[:, gate_mocap_ids].set(gate_quat))
        data = data.replace(
            mocap_pos=data.mocap_pos.at[:, obstacle_mocap_ids].set(nominal_obstacle_pos)
        )
        keys = jax.random.split(key, len(randomization_fns))
        for key, randomize_fn in zip(keys, randomization_fns, strict=True):
            data = randomize_fn(data, mask, key)
        return data

    return track_randomization


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance
