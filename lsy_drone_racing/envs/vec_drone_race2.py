"""Collection of environments for multi-drone racing simulations.

This module is the multi-drone counterpart to the regular drone racing environments.
"""

from __future__ import annotations

import copy as copy
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import gymnasium
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from crazyflow import Sim
from crazyflow.sim.symbolic import symbolic_attitude
from crazyflow.utils import leaf_replace
from flax.struct import dataclass
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax.scipy.spatial.transform import Rotation as JaxR
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

if TYPE_CHECKING:
    from crazyflow.sim.structs import SimData
    from jax import Array
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region EnvData


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment."""

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    marked_for_reset: Array
    disabled_drones: Array
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    rpy_limit_low: Array
    rpy_limit_high: Array
    gate_mocap_ids: Array

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_drones: int,
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        gate_mocap_ids: Array,
        device: jax.Device,
    ) -> EnvData:
        """Create a new environment data struct with default values."""
        return cls(
            target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jp.zeros((n_envs, n_drones, n_gates), dtype=bool, device=device),
            obstacles_visited=jp.zeros((n_envs, n_drones, n_obstacles), dtype=bool, device=device),
            last_drone_pos=jp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            marked_for_reset=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            disabled_drones=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jp.array(contact_masks, dtype=bool, device=device),
            pos_limit_low=jp.full(3, -jp.inf, dtype=np.float32, device=device),
            pos_limit_high=jp.full(3, jp.inf, dtype=np.float32, device=device),
            rpy_limit_low=jp.full(3, -jp.pi, dtype=np.float32, device=device),
            rpy_limit_high=jp.full(3, jp.pi, dtype=np.float32, device=device),
            gate_mocap_ids=jp.array(gate_mocap_ids, dtype=int, device=device),
        )


# region Core Env


class VectorMultiDroneRaceEnv(gymnasium.Env):
    """A Gymnasium environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.

    The environment provides:
    - A customizable track with gates and obstacles
    - Configurable simulation and control frequencies
    - Support for different physics models (e.g., identified dynamics, analytical dynamics)
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
        n_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the DroneRacingEnv.

        Args:
            n_drones: Number of drones in the environment.
            freq: Environment frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            random_resets: Flag to randomize the environment on reset.
            seed: Random seed of the environment.
        """
        super().__init__()
        self.sim = Sim(
            n_worlds=n_envs,
            n_drones=n_drones,
            physics=sim_config.physics,
            control=sim_config.get("control", "state"),
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=seed,
            device=device,
        )
        # Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")

        # Env settings
        self.freq = freq
        self.seed = seed
        self.autoreset = True  # Can be overridden by subclasses
        self.device = jax.devices(device)[0]
        self.symbolic = symbolic_attitude(1 / self.freq)
        self.random_resets = random_resets
        self.sensor_range = sensor_range
        self.gates, self.obstacles, self.drone = self.load_track(track)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        self.randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Spaces
        self.single_action_space = spaces.Box(low=-1, high=1, shape=(n_drones, 13))
        self.action_space = batch_space(self.single_action_space, n_envs)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_drones, 3)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(n_drones, 3)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(n_drones, 3)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(n_drones, 3)),
                "target_gate": spaces.MultiDiscrete([n_gates] * n_drones, start=[-1] * n_drones),
                "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_drones, n_gates, 3)),
                "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(n_drones, n_gates, 3)),
                "gates_visited": spaces.Box(low=0, high=1, shape=(n_drones, n_gates), dtype=bool),
                "obstacles_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_drones, n_obstacles, 3)
                ),
                "obstacles_visited": spaces.Box(
                    low=0, high=1, shape=(n_drones, n_obstacles), dtype=bool
                ),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, n_envs)

        # Compile the reset and step functions with custom hooks
        self.setup_sim()

        # Create the environment data struct
        rpy_limit = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)
        masks = self.load_contact_masks()
        gate_mocap_ids = self.gates["mocap_ids"]
        self.data = EnvData.create(
            n_envs, n_drones, n_gates, n_obstacles, masks, gate_mocap_ids, self.device
        )
        self.data = self.data.replace(
            pos_limit_low=np.array([-3, -3, 0]),
            pos_limit_high=np.array([3, 3, 2.5]),
            rpy_limit_low=-rpy_limit,
            rpy_limit_high=rpy_limit,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        mask: NDArray[np.bool_] | None = None,
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Observation and info.
        """
        # TODO: Allow per-world sim seeding
        if seed is not None:
            self.sim.seed(seed)
        elif not self.random_resets:
            self.sim.seed(self.seed)
        # Randomization of gates, obstacles and drones is compiled into the sim reset function with
        # the sim.reset_hook function, so we don't need to explicitly do it here
        self.sim.reset(mask=mask)
        self.data = self.reset_data(self.data, self.sim.data.states.pos, mask)
        return self.obs(), self.info()

    @staticmethod
    def reset_data(data: EnvData, drone_pos: Array, mask: NDArray[np.bool_]) -> EnvData:
        """Reset auxiliary variables of the environment data."""
        target_gate = data.target_gate.at[mask, ...].set(0)
        last_drone_pos = data.last_drone_pos.at[mask, ...].set(drone_pos)
        disabled_drones = data.disabled_drones.at[mask, ...].set(False)
        return data.replace(
            target_gate=target_gate, last_drone_pos=last_drone_pos, disabled_drones=disabled_drones
        )

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
        self.apply_action(action)
        self.sim.step(self.sim.freq // self.freq)
        self.sim.data = self.warp_disabled_drones(self.sim.data, self.data.disabled_drones)
        # Apply the environment logic. Check which drones are now disabled, check which gates have
        # been passed, and update the target gate.
        drone_pos, drone_quat = self.sim.data.states.pos, self.sim.data.states.quat
        gate_pos, gate_quat = self.sim.data.mjx_data.mocap_pos, self.sim.data.mjx_data.mocap_quat
        contacts = self.sim.contacts()
        self.data = self._step(self.data, drone_pos, drone_quat, gate_pos, gate_quat, contacts)
        # Auto-reset envs. Add configuration option to disable for single-world envs
        if self.autoreset and self.data.marked_for_reset.any():
            self.reset(mask=self.marked_for_reset)
        terminated, truncated = self.terminated(), self.truncated()
        self.marked_for_reset = np.any(terminated | truncated, axis=-1)
        return self.obs(), self.reward(), terminated, truncated, self.info()

    @staticmethod
    def _step(
        data: EnvData,
        drone_pos: Array,
        drone_quat: Array,
        gates_pos: Array,
        gates_quat: Array,
        contacts: Array,
    ) -> EnvData:
        """Step the environment data."""
        disabled_drones = _disabled_drones(drone_pos, drone_quat, contacts, data)
        gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], data.gate_mocap_ids]
        gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], data.gate_mocap_ids]
        # We need to convert the gate quat from MuJoCo order to scipy order
        gate_quat = gate_quat[..., [3, 0, 1, 2]]
        passed = _gate_passed(drone_pos, gate_pos, gate_quat, data.last_drone_pos)
        target_gate = data.target_gate + passed * ~disabled_drones
        target_gate = jp.where(target_gate >= data.n_gates, -1, target_gate)
        marked_for_reset = ...  # TODO: Implement this
        data = data.replace(
            last_drone_pos=drone_pos, target_gate=target_gate, disabled_drones=disabled_drones
        )
        return data

    @staticmethod
    @jax.jit
    def _marked_for_reset(terminated: Array, truncated: Array) -> Array:
        """Mark the drones for reset if they are terminated or truncated."""
        return jp.any(terminated | truncated, axis=-1)

    def apply_action(self, action: NDArray[np.floating]):
        """Apply the commanded state action to the simulation."""
        action = action.reshape((self.sim.n_worlds, self.sim.n_drones, 13))
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        self.sim.state_control(action)

    def render(self):
        """Render the environment."""
        self.sim.render()

    def obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the observation of the environment."""
        # TODO: Accelerate this function
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        self.gates_visited, gates_pos, gates_rpy = self._obs_gates(
            self.data.gates_visited,
            self.sim.data.states.pos,
            self.sim.data.mjx_data.mocap_pos,
            self.sim.data.mjx_data.mocap_quat,
            self.gates["mocap_ids"],
            self.sensor_range,
            self.gates["nominal_pos"],
            self.gates["nominal_rpy"],
        )
        self.obstacles_visited, obstacles_pos = self._obs_obstacles(
            self.data.obstacles_visited,
            self.sim.data.states.pos,
            self.sim.data.mjx_data.mocap_pos,
            self.obstacles["mocap_ids"],
            self.sensor_range,
            self.obstacles["nominal_pos"],
        )
        quat = self.sim.data.states.quat
        rpy = R.from_quat(quat.reshape(-1, 4)).as_euler("xyz").reshape((*quat.shape[:-1], 3))
        # TODO: Decide on observation disturbances
        obs = {
            "pos": np.array(self.sim.data.states.pos, dtype=np.float32),
            "rpy": rpy.astype(np.float32),
            "vel": np.array(self.sim.data.states.vel, dtype=np.float32),
            "ang_vel": np.array(self.sim.data.states.rpy_rates, dtype=np.float32),
            "target_gate": self.target_gate,
            "gates_pos": np.asarray(gates_pos, dtype=np.float32),
            "gates_rpy": np.asarray(gates_rpy, dtype=np.float32),
            "gates_visited": np.asarray(self.gates_visited, dtype=bool),
            "obstacles_pos": np.asarray(obstacles_pos, dtype=np.float32),
            "obstacles_visited": np.asarray(self.obstacles_visited, dtype=bool),
        }
        return obs

    @staticmethod
    @jax.jit
    def _obs_gates(
        visited: Array,
        drone_pos: Array,
        mocap_pos: Array,
        mocap_quat: Array,
        mocap_ids: Array,
        sensor_range: float,
        nominal_pos: NDArray,
        nominal_rpy: NDArray,
    ) -> tuple[Array, Array, Array]:
        """Get the nominal or real gate positions and orientations depending on the sensor range."""
        real_pos = mocap_pos[:, mocap_ids]
        real_rpy = JaxR.from_quat(mocap_quat[:, mocap_ids][..., [1, 2, 3, 0]]).as_euler("xyz")
        dpos = drone_pos[..., None, :2] - real_pos[:, None, :, :2]
        visited = jp.logical_or(visited, jp.linalg.norm(dpos, axis=-1) < sensor_range)
        gates_pos = jp.where(visited[..., None], real_pos[:, None], nominal_pos[None, None])
        gates_rpy = jp.where(visited[..., None], real_rpy[:, None], nominal_rpy[None, None])
        return visited, gates_pos, gates_rpy

    @staticmethod
    @jax.jit
    def _obs_obstacles(
        visited: NDArray,
        drone_pos: Array,
        mocap_pos: Array,
        mocap_ids: NDArray,
        sensor_range: float,
        nominal_pos: NDArray,
    ) -> tuple[Array, Array]:
        real_pos = mocap_pos[:, mocap_ids]
        dpos = drone_pos[..., None, :2] - real_pos[:, None, :, :2]
        visited = jp.logical_or(visited, jp.linalg.norm(dpos, axis=-1) < sensor_range)
        obstacles_pos = jp.where(visited[..., None], real_pos[:, None], nominal_pos[None, None])
        return visited, obstacles_pos

    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 * (self.target_gate == -1)

    def terminated(self) -> bool:
        """Check if the episode is terminated.

        Returns:
            True if all drones have been disabled, else False.
        """
        return self.disabled_drones

    def truncated(self) -> NDArray[np.bool_]:
        """Array of booleans indicating if the episode is truncated."""
        return np.zeros((self.sim.n_worlds, self.sim.n_drones), dtype=np.bool_)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        contacts, mask = self.sim.contacts(), self.contact_masks
        return {"collisions": self._collision_info(contacts, mask), "symbolic_model": self.symbolic}

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @staticmethod
    @jax.jit
    def _collision_info(contacts: Array, mask: Array) -> Array:
        return jp.any(jp.logical_and(contacts[:, None, :], mask), axis=-1)

    def load_track(self, track: dict) -> tuple[dict, dict, dict]:
        """Load the track from the config file."""
        gate_pos = np.array([g["pos"] for g in track.gates])
        gate_rpy = np.array([g["rpy"] for g in track.gates])
        gates = {"pos": gate_pos, "rpy": gate_rpy, "nominal_pos": gate_pos, "nominal_rpy": gate_rpy}
        obstacle_pos = np.array([o["pos"] for o in track.obstacles])
        obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos}
        drone_keys = ("pos", "rpy", "vel", "rpy_rates")
        drone = {k: np.array(track.drone.get(k), dtype=np.float32) for k in drone_keys}
        drone["quat"] = R.from_euler("xyz", drone["rpy"]).as_quat()
        return gates, obstacles, drone

    def load_contact_masks(self) -> Array:
        """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        object_contacts = n_obstacles + n_gates * 5 + 1  # 5 geoms per gate, 1 for the floor
        drone_contacts = (self.sim.n_drones - 1) * self.sim.n_drones // 2
        n_contacts = self.sim.n_drones * object_contacts + drone_contacts
        masks = np.zeros((self.sim.n_drones, n_contacts), dtype=bool)
        mj_model = self.sim.mj_model
        geom1 = self.sim.data.mjx_data.contact.geom1[0]  # We only need one world to create the mask
        geom2 = self.sim.data.mjx_data.contact.geom2[0]
        for i in range(self.sim.n_drones):
            geom_start = mj_model.body_geomadr[mj_model.body(f"drone:{i}").id]
            geom_count = mj_model.body_geomnum[mj_model.body(f"drone:{i}").id]
            geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
            geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
            masks[i, :] = geom1_valid | geom2_valid
        geom_start = mj_model.body_geomadr[mj_model.body("world").id]
        geom_count = mj_model.body_geomnum[mj_model.body("world").id]
        geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
        geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
        masks[:, (geom1_valid | geom2_valid).squeeze()] = 0  # Floor contacts are not collisions
        masks = np.tile(masks[None, ...], (self.sim.n_worlds, 1, 1))
        return jp.array(masks, dtype=bool, device=self.device)

    def setup_sim(self):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        self._load_track_into_sim(self.gates, self.obstacles)
        pos = self.sim.data.states.pos.at[...].set(self.drone["pos"])
        quat = self.sim.data.states.quat.at[...].set(self.drone["quat"])
        vel = self.sim.data.states.vel.at[...].set(self.drone["vel"])
        rpy_rates = self.sim.data.states.rpy_rates.at[...].set(self.drone["rpy_rates"])
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
        mocap_ids = [int(mj_model.body(f"gate:{i}").mocapid) for i in range(n_gates)]
        gates["mocap_ids"] = jp.array(mocap_ids, dtype=np.int32, device=self.device)
        obstacles["ids"] = [mj_model.body(f"obstacle:{i}").id for i in range(n_obstacles)]
        mocap_ids = [int(mj_model.body(f"obstacle:{i}").mocapid) for i in range(n_obstacles)]
        obstacles["mocap_ids"] = jp.array(mocap_ids, dtype=np.int32, device=self.device)

    @staticmethod
    @jax.jit
    def warp_disabled_drones(data: SimData, mask: NDArray) -> SimData:
        """Warp the disabled drones below the ground."""
        pos = jax.numpy.where(mask[..., None], -1, data.states.pos)
        return data.replace(states=data.states.replace(pos=pos))

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()


# region Env fns


def _disabled_drones(pos: Array, quat: Array, contacts: Array, data: EnvData) -> Array:
    rpy = JaxR.from_quat(quat).as_euler("xyz")
    disabled = jp.logical_or(data.disabled_drones, jp.all(pos < data.pos_limit_low, axis=-1))
    disabled = jp.logical_or(disabled, jp.all(pos > data.pos_limit_high, axis=-1))
    disabled = jp.logical_or(disabled, jp.all(rpy < data.rpy_limit_low, axis=-1))
    disabled = jp.logical_or(disabled, jp.all(rpy > data.rpy_limit_high, axis=-1))
    disabled = jp.logical_or(disabled, data.target_gate == -1)
    contacts = jp.any(jp.logical_and(contacts[:, None, :], data.contact_masks), axis=-1)
    disabled = jp.logical_or(disabled, contacts)
    return disabled


def _gate_passed(
    gate_pos: Array, gate_quat: Array, drone_pos: Array, last_drone_pos: NDArray
) -> bool:
    """Check if the drone has passed a gate.

    Returns:
        True if the drone has passed a gate, else False.
    """
    # TODO: Test. Cover cases with no gates.
    gate_rot = JaxR.from_quat(gate_quat)
    gate_size = (0.45, 0.45)
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check if the line between the last position and the current position intersects the plane.
    # If so, calculate the point of the intersection and check if it is within the gate box.
    passed_plane = (last_pos_local[..., 1] < 0) & (pos_local[..., 1] > 0)
    alpha = -last_pos_local[..., 1] / (pos_local[..., 1] - last_pos_local[..., 1])
    x_intersect = alpha * (pos_local[..., 0]) + (1 - alpha) * last_pos_local[..., 0]
    z_intersect = alpha * (pos_local[..., 2]) + (1 - alpha) * last_pos_local[..., 2]
    in_box = (abs(x_intersect) < gate_size[0] / 2) & (abs(z_intersect) < gate_size[1] / 2)
    return passed_plane & in_box


# region AttitudeEnv


class VectorDroneRaceAttitudeEnv(VectorMultiDroneRaceEnv):
    """Drone racing environment with a collective thrust attitude command interface.

    The action space consists of the collective thrust and body-fixed attitude commands
    [collective_thrust, roll, pitch, yaw].
    """

    def __init__(
        self,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        random_resets: bool = False,
        seed: int = 1337,
    ):
        """Initialize the DroneRacingAttitudeEnv.

        Args:
            n_drones: Number of drones in the environment.
            freq: Environment frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            random_resets: Flag to randomize the environment on reset.
            seed: Random seed of the environment.
        """
        sim_config.control = "attitude"
        super().__init__(
            n_drones,
            freq,
            sim_config,
            sensor_range,
            track,
            disturbances,
            randomizations,
            random_resets,
            seed,
        )
        bounds = np.array([[1, np.pi, np.pi, np.pi] for _ in range(n_drones)], dtype=np.float32)
        self.action_space = spaces.Box(low=-bounds, high=bounds)

    def apply_action(self, action: NDArray[np.floating]):
        """Apply the commanded attitude action to the simulation."""
        action = action.reshape((self.sim.n_worlds, self.sim.n_drones, 4))
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        self.sim.attitude_control(action)


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
