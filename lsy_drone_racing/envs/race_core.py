"""Collection of environments for multi-drone racing simulations.

This module is the multi-drone counterpart to the regular drone racing environments.
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
from crazyflow import Sim
from crazyflow.sim.symbolic import symbolic_attitude
from flax.struct import dataclass
from gymnasium import spaces
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
    from crazyflow.sim.symbolic import SymbolicModel
    from jax import Array, Device
    from ml_collections import ConfigDict
    from mujoco import MjSpec
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
    steps: Array
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    rpy_limit_low: Array
    rpy_limit_high: Array
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
        rpy_limit_low: Array,
        rpy_limit_high: Array,
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
            rpy_limit_low=jp.array(rpy_limit_low, dtype=np.float32, device=device),
            rpy_limit_high=jp.array(rpy_limit_high, dtype=np.float32, device=device),
            gate_mj_ids=jp.array(gate_mj_ids, dtype=int, device=device),
            obstacle_mj_ids=jp.array(obstacle_mj_ids, dtype=int, device=device),
            max_episode_steps=jp.array([max_episode_steps], dtype=int, device=device),
            sensor_range=jp.array([sensor_range], dtype=jp.float32, device=device),
        )


def action_space(control_mode: Literal["state", "attitude"]) -> spaces.Box:
    if control_mode == "state":
        return spaces.Box(low=-1, high=1, shape=(13,))
    elif control_mode == "attitude":
        lim = np.array([1, np.pi, np.pi, np.pi])
        return spaces.Box(low=-lim, high=lim)
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")


def observation_space(n_gates: int, n_obstacles: int) -> spaces.Dict:
    """Create the observation space for the environment."""
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "target_gate": spaces.MultiDiscrete([n_gates], start=[-1]),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(n_gates, 3)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
    }
    return spaces.Dict(obs_spec)


# region Core Env


class RaceCoreEnv:
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
        max_episode_steps: int = 1500,
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
        self.random_resets = random_resets
        self.sensor_range = sensor_range
        self.gates, self.obstacles, self.drone = self._load_track(track)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        self.randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Load the track into the simulation and compile the reset and step functions with hooks
        self._setup_sim()

        # Create the environment data struct. TODO: Remove unnecessary replacements
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        masks = self._load_contact_masks(self.sim)
        masks = jp.array(masks, dtype=bool, device=self.device)
        gate_mj_ids, obstacle_mj_ids = self.gates["mj_ids"], self.obstacles["mj_ids"]
        pos_limit_low = jp.array([-3, -3, 0], dtype=np.float32, device=self.device)
        pos_limit_high = jp.array([3, 3, 2.5], dtype=np.float32, device=self.device)
        rpy_limit = jp.array([jp.pi / 2, jp.pi / 2, jp.pi], dtype=jp.float32, device=self.device)
        self.data = EnvData.create(
            n_envs,
            n_drones,
            n_gates,
            n_obstacles,
            masks,
            gate_mj_ids,
            obstacle_mj_ids,
            max_episode_steps,
            sensor_range,
            pos_limit_low,
            pos_limit_high,
            -rpy_limit,
            rpy_limit,
            self.device,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            mask: Mask of worlds to reset.

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
        self.data = self._reset(self.data, self.sim.data.states.pos, mask)
        return self.obs(), self.info()

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
        # Warp drones that have crashed outside the track to prevent them from interfering with
        # other drones still in the race
        self.sim.data = self._warp_disabled_drones(self.sim.data, self.data.disabled_drones)
        # Apply the environment logic. Check which drones are now disabled, check which gates have
        # been passed, and update the target gate.
        drone_pos, drone_quat = self.sim.data.states.pos, self.sim.data.states.quat
        mocap_pos, mocap_quat = self.sim.data.mjx_data.mocap_pos, self.sim.data.mjx_data.mocap_quat
        contacts = self.sim.contacts()
        # Get marked_for_reset before it is updated, because the autoreset needs to be based on the
        # previous flags, not the ones from the current step
        marked_for_reset = self.data.marked_for_reset
        # Apply the environment logic with updated simulation data.
        self.data = self._step(self.data, drone_pos, drone_quat, mocap_pos, mocap_quat, contacts)
        # Auto-reset envs. Add configuration option to disable for single-world envs
        if self.autoreset and marked_for_reset.any():
            self.reset(mask=marked_for_reset)
        return self.obs(), self.reward(), self.terminated(), self.truncated(), self.info()

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

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()

    def obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the observation of the environment."""
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        gates_pos, gates_rpy, obstacles_pos = self._obs(
            self.sim.data.mjx_data.mocap_pos,
            self.sim.data.mjx_data.mocap_quat,
            self.data.gates_visited,
            self.gates["mj_ids"],
            self.gates["nominal_pos"],
            self.gates["nominal_rpy"],
            self.data.obstacles_visited,
            self.obstacles["mj_ids"],
            self.obstacles["nominal_pos"],
        )
        quat = self.sim.data.states.quat
        rpy = R.from_quat(quat.reshape(-1, 4)).as_euler("xyz").reshape((*quat.shape[:-1], 3))
        obs = {
            "pos": np.array(self.sim.data.states.pos, dtype=np.float32),
            "rpy": rpy.astype(np.float32),
            "vel": np.array(self.sim.data.states.vel, dtype=np.float32),
            "ang_vel": np.array(self.sim.data.states.ang_vel, dtype=np.float32),
            "target_gate": np.array(self.data.target_gate, dtype=int),
            "gates_pos": np.asarray(gates_pos, dtype=np.float32),
            "gates_rpy": np.asarray(gates_rpy, dtype=np.float32),
            "gates_visited": np.asarray(self.data.gates_visited, dtype=bool),
            "obstacles_pos": np.asarray(obstacles_pos, dtype=np.float32),
            "obstacles_visited": np.asarray(self.data.obstacles_visited, dtype=bool),
        }
        return obs

    def reward(self) -> NDArray[np.float32]:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return np.array(-1.0 * (self.data.target_gate == -1), dtype=np.float32)

    def terminated(self) -> NDArray[np.bool_]:
        """Check if the episode is terminated.

        Returns:
            True if all drones have been disabled, else False.
        """
        return np.array(self.data.disabled_drones, dtype=bool)

    def truncated(self) -> NDArray[np.bool_]:
        """Array of booleans indicating if the episode is truncated."""
        return np.tile(self.data.steps >= self.data.max_episode_steps, (self.sim.n_drones, 1))

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @property
    def symbolic_model(self) -> SymbolicModel:
        """The symbolic model of the environment."""
        return symbolic_attitude(1 / self.freq)

    @staticmethod
    @jax.jit
    def _reset(data: EnvData, drone_pos: Array, mask: Array | None = None) -> EnvData:
        """Reset auxiliary variables of the environment data."""
        mask = jp.ones(data.steps.shape, dtype=bool) if mask is None else mask
        target_gate = jp.where(mask[..., None], 0, data.target_gate)
        last_drone_pos = jp.where(mask[..., None, None], drone_pos, data.last_drone_pos)
        disabled_drones = jp.where(mask[..., None], False, data.disabled_drones)
        steps = jp.where(mask, 0, data.steps)
        gates_visited = jp.where(mask[..., None, None], False, data.gates_visited)
        obstacles_visited = jp.where(mask[..., None, None], False, data.obstacles_visited)
        data = data.replace(
            target_gate=target_gate,
            last_drone_pos=last_drone_pos,
            disabled_drones=disabled_drones,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            steps=steps,
        )
        return data

    @staticmethod
    @jax.jit
    def _step(
        data: EnvData,
        drone_pos: Array,
        drone_quat: Array,
        mocap_pos: Array,
        mocap_quat: Array,
        contacts: Array,
    ) -> EnvData:
        """Step the environment data."""
        n_gates = len(data.gate_mj_ids)
        disabled_drones = RaceCoreEnv._disabled_drones(drone_pos, drone_quat, contacts, data)
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        # We need to convert the mocap quat from MuJoCo order to scipy order
        gates_quat = mocap_quat[:, data.gate_mj_ids][..., [3, 0, 1, 2]]
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        # Extract the gate poses of the current target gates and check if the drones have passed
        # them between the last and current position
        gate_ids = data.gate_mj_ids[data.target_gate % n_gates]
        gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], gate_ids]
        gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], gate_ids]
        passed = RaceCoreEnv._gate_passed(drone_pos, data.last_drone_pos, gate_pos, gate_quat)
        # Update the target gate index. Increment by one if drones have passed a gate
        target_gate = data.target_gate + passed * ~disabled_drones
        target_gate = jp.where(target_gate >= n_gates, -1, target_gate)
        steps = data.steps + 1
        truncated = steps >= data.max_episode_steps
        marked_for_reset = jp.all(disabled_drones | truncated[..., None], axis=-1)
        # Update which gates and obstacles are or have been in range of the drone
        sensor_range = data.sensor_range
        gates_visited = RaceCoreEnv._visited(drone_pos, gates_pos, sensor_range, data.gates_visited)
        obstacles_visited = RaceCoreEnv._visited(
            drone_pos, obstacles_pos, sensor_range, data.obstacles_visited
        )
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
        nominal_gate_rpy: NDArray,
        obstacles_visited: Array,
        obstacle_mocap_ids: Array,
        nominal_obstacle_pos: NDArray,
    ) -> tuple[Array, Array]:
        """Get the nominal or real gate positions and orientations depending on the sensor range."""
        mask, real_pos = gates_visited[..., None], mocap_pos[:, gate_mocap_ids]
        real_rpy = JaxR.from_quat(mocap_quat[:, gate_mocap_ids][..., [1, 2, 3, 0]]).as_euler("xyz")
        gates_pos = jp.where(mask, real_pos[:, None], nominal_gate_pos[None, None])
        gates_rpy = jp.where(mask, real_rpy[:, None], nominal_gate_rpy[None, None])
        mask, real_pos = obstacles_visited[..., None], mocap_pos[:, obstacle_mocap_ids]
        obstacles_pos = jp.where(mask, real_pos[:, None], nominal_obstacle_pos[None, None])
        return gates_pos, gates_rpy, obstacles_pos

    def _load_track(self, track: dict) -> tuple[dict, dict, dict]:
        """Load the track from the config file."""
        gate_pos = np.array([g["pos"] for g in track.gates])
        gate_rpy = np.array([g["rpy"] for g in track.gates])
        gates = {"pos": gate_pos, "rpy": gate_rpy, "nominal_pos": gate_pos, "nominal_rpy": gate_rpy}
        obstacle_pos = np.array([o["pos"] for o in track.obstacles])
        obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos}
        drone_keys = ("pos", "rpy", "vel", "ang_vel")
        drone = {k: np.array(track.drone.get(k), dtype=np.float32) for k in drone_keys}
        drone["quat"] = R.from_euler("xyz", drone["rpy"]).as_quat()
        return gates, obstacles, drone

    def _setup_sim(self):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        self._load_track_into_sim(gate_spec, obstacle_spec)
        self._register_object_ids()
        # Set the initial drone states
        pos = self.sim.data.states.pos.at[...].set(self.drone["pos"])
        quat = self.sim.data.states.quat.at[...].set(self.drone["quat"])
        vel = self.sim.data.states.vel.at[...].set(self.drone["vel"])
        ang_vel = self.sim.data.states.ang_vel.at[...].set(self.drone["ang_vel"])
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        self.sim.data = self.sim.data.replace(states=states)
        # Build the reset randomizations and disturbances into the sim itself
        self.sim.reset_hook = build_reset_hook(
            self.randomizations, self.gates["mj_ids"], self.obstacles["mj_ids"]
        )
        if "dynamics" in self.disturbances:
            self.sim.disturbance_fn = build_dynamics_disturbance_fn(self.disturbances["dynamics"])
        # Save the reset state and rebuild the reset function
        self.sim.build(mjx=False, data=False)

    def _load_track_into_sim(self, gate_spec: MjSpec, obstacle_spec: MjSpec):
        """Load the track into the simulation."""
        frame = self.sim.spec.worldbody.add_frame()
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        for i in range(n_gates):
            gate = frame.attach_body(gate_spec.find_body("gate"), "", f":{i}")
            gate.pos = self.gates["pos"][i]
            # Convert from scipy order to MuJoCo order
            gate.quat = R.from_euler("xyz", self.gates["rpy"][i]).as_quat()[[3, 0, 1, 2]]
            gate.mocap = True  # Make mocap to modify the position of static bodies during sim
        for i in range(n_obstacles):
            obstacle = frame.attach_body(obstacle_spec.find_body("obstacle"), "", f":{i}")
            obstacle.pos = self.obstacles["pos"][i]
            obstacle.mocap = True
        self.sim.build(data=False, default_data=False)

    def _register_object_ids(self) -> tuple[dict, dict]:
        """Register the ids and mocap ids of the gates and obstacles."""
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        mj_model = self.sim.mj_model
        self.gates["ids"] = [mj_model.body(f"gate:{i}").id for i in range(n_gates)]
        mj_ids = [int(mj_model.body(f"gate:{i}").mocapid) for i in range(n_gates)]
        self.gates["mj_ids"] = jp.array(mj_ids, dtype=np.int32, device=self.device)
        self.obstacles["ids"] = [mj_model.body(f"obstacle:{i}").id for i in range(n_obstacles)]
        mj_ids = [int(mj_model.body(f"obstacle:{i}").mocapid) for i in range(n_obstacles)]
        self.obstacles["mj_ids"] = jp.array(mj_ids, dtype=np.int32, device=self.device)

    def _load_contact_masks(self, sim: Sim) -> Array:
        """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
        n_contacts = len(self.sim.data.mjx_data.contact.geom1[0])
        masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)
        mj_model = sim.mj_model
        geom1 = sim.data.mjx_data.contact.geom1[0]  # We only need one world to create the mask
        geom2 = sim.data.mjx_data.contact.geom2[0]
        for i in range(sim.n_drones):
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
        masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
        return masks

    @staticmethod
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

    @staticmethod
    def _gate_passed(
        drone_pos: Array, last_drone_pos: Array, gate_pos: Array, gate_quat: Array
    ) -> bool:
        """Check if the drone has passed a gate.

        Returns:
            True if the drone has passed a gate, else False.
        """
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

    @staticmethod
    def _visited(drone_pos: Array, target_pos: Array, sensor_range: float, visited: Array) -> Array:
        dpos = drone_pos[..., None, :2] - target_pos[:, None, :, :2]
        return jp.logical_or(visited, jp.linalg.norm(dpos, axis=-1) < sensor_range)

    @staticmethod
    @jax.jit
    def _warp_disabled_drones(data: SimData, mask: Array) -> SimData:
        """Warp the disabled drones below the ground."""
        pos = jax.numpy.where(mask[..., None], -1, data.states.pos)
        return data.replace(states=data.states.replace(pos=pos))


# region AttitudeEnv


class VectorDroneRaceAttitudeEnv(RaceCoreEnv):
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
