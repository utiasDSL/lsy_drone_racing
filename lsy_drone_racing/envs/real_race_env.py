from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import jax
import numpy as np
import rclpy
from gymnasium import Env
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray

from lsy_drone_racing.envs.utils import load_track
from lsy_drone_racing.ros import ROSConnector
from lsy_drone_racing.ros.ros_utils import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.utils import gate_passed

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment."""

    target_gate: NDArray
    gates_visited: NDArray
    obstacles_visited: NDArray
    last_drone_pos: NDArray[np.float32]

    @classmethod
    def create(cls, n_drones: int, n_gates: int, n_obstacles: int) -> EnvData:
        """Create an instance of the EnvData class."""
        return EnvData(
            target_gate=np.zeros(n_drones, dtype=int),
            gates_visited=np.zeros((n_drones, n_gates), dtype=bool),
            obstacles_visited=np.zeros((n_drones, n_obstacles), dtype=bool),
            last_drone_pos=np.zeros((n_drones, 3), dtype=np.float32),
        )


class RealRaceCoreEnv:
    """Deployable version of the multi-agent drone racing environment."""

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        n_drones: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the deployable version of the multi-agent drone racing environment.

        Args:
            n_drones: Number of drones.
            freq: Environment step frequency.
            sensor_range: Sensor range.
        """
        rclpy.init()
        # Static env data
        self.n_drones = n_drones
        self.gates, self.obstacles, self.drones = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array([-3.0, -3.0, 0.0])
        self.pos_limit_high = np.array([3.0, 3.0, 2.5])
        self.sensor_range = sensor_range
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        self.channel = drones[rank]["channel"]
        self.rank = rank
        self.freq = freq
        self.device = jax.devices("cpu")[0]
        assert control_mode in ["state", "attitude"], f"Invalid control mode {control_mode}"
        self.control_mode = control_mode
        self.randomizations = randomizations
        # Dynamic data
        self.data = EnvData.create(
            n_drones=n_drones, n_gates=self.n_gates, n_obstacles=self.n_obstacles
        )

        self._ros_connector = ROSConnector(estimator_names=self.drone_names, timeout=5.0)
        post_fix = "full_state" if control_mode == "attitude" else "state"
        msg_name = f"/{self.drone_name}/" + post_fix
        self.node = rclpy.create_node("RealRaceCoreEnv" + uuid4().hex)
        self._action_pub = self.node.create_publisher(Float64MultiArray, msg_name, 10)
        self._jit()

    def _reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        if options is None or options.get("check_race_track", True):
            check_race_track(self.gates, self.obstacles, self.randomizations)
        if options is None or options.get("check_drone_start_pos", True):
            check_drone_start_pos(self.drones.pos, self.randomizations, self.drone_name)
        self._reset_env_data(self.data)
        # Update the ground truth position and orientation of the gates and obstacles
        tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
        tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
        ros_connector = ROSConnector(tf_names=tf_names, timeout=5.0)
        for i in range(self.n_gates):
            self.gates.pos[i, ...] = ros_connector.pos[f"gate{i + 1}"]
            self.gates.quat[i, ...] = ros_connector.quat[f"gate{i + 1}"]
        for i in range(self.n_obstacles):
            self.obstacles.pos[i, ...] = ros_connector.pos[f"obstacle{i + 1}"]
        ros_connector.close()
        return self.obs(), self.info()

    def _step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment."""
        # Note: We do not send the action to the drone here.
        self.send_action(action)
        self._action_pub.publish(Float64MultiArray(data=action))
        obs = self.obs()
        gate_pos = self.gates.pos[self.data.target_gate]
        gate_quat = self.gates.quat[self.data.target_gate]
        drone_pos = obs["pos"]

        with jax.default_device(self.device):  # Ensure gate_passed runs on the CPU
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45)
            )
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos
        return obs, self.reward(), self.terminated(), self.truncated(), self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the observation of the environment."""
        # If gates/obstacles are in sensor range use the actual pose, otherwise use the nominal pose
        # The actual pose is measured at the beginning of the episode and is not updated during the
        # episode. If we want to use dynamic gates/obstacles, we need to update the poses here.
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
        drone_pos = np.stack(
            [self._ros_connector.pos[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_quat = np.stack(
            [self._ros_connector.quat[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_vel = np.stack(
            [self._ros_connector.vel[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_ang_vel = np.stack(
            [self._ros_connector.ang_vel[drone] for drone in self.drone_names], dtype=np.float32
        )
        obs = {
            "pos": drone_pos,
            "quat": drone_quat,
            "vel": drone_vel,
            "ang_vel": drone_ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
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
        return -1.0 * (self.data.target_gate == -1)  # Implicit float conversion

    def terminated(self) -> bool:
        """Check if the episode is terminated."""
        return self.data.target_gate == -1

    def truncated(self) -> bool:
        """Check if the episode is truncated."""
        return False

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        if self.control_mode == "attitude":
            self.drone.commander.send_setpoint(*action)
        else:
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            # TODO: We currently limit ourselves to yaw rotation only because the simulation is
            # based on the old crazyswarm full_state command definition. Once the simulation does
            # support the real full_state command, we can remove this limitation and use full
            # quaternions as inputs
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            return
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )
            self._ros_connector.send_action(self.drone_name, action)

    def _reset_env_data(self, data: EnvData):
        """Reset the environment data."""
        data.target_gate[...] = 0
        data.gates_visited[...] = False
        data.obstacles_visited[...] = False
        drone_pos = np.stack([self._ros_connector.pos[n] for n in self.drone_names])
        data.last_drone_pos[...] = drone_pos

    def _jit(self):
        """JIT compile jax functions.

        We compile all jit-compiled functions at startup to avoid the overhead of compiling them
        at the first call when the drone is already in the air.
        """
        drone_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_quat = np.zeros((self.n_drones, 4), dtype=np.float32)
        with jax.default_device(self.device):
            jax.block_until_ready(
                gate_passed(drone_pos, drone_pos, gate_pos, gate_quat, (0.45, 0.45))
            )

    def close(self):
        """Close the environment."""
        self._ros_connector.close()
        self._action_pub.destroy()
        self.node.destroy_node()


class RealDroneRaceEnv(RealRaceCoreEnv, Env):
    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        super().__init__(
            drones=drones,
            rank=rank,
            n_drones=1,
            freq=freq,
            track=track,
            randomizations=randomizations,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        return self._reset(seed=seed, options=options)

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        return self._step(action)


class RealMultiDroneRaceEnv(RealRaceCoreEnv, Env):
    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        n_drones: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        super().__init__(
            drones=drones,
            rank=rank,
            n_drones=n_drones,
            freq=freq,
            randomizations=randomizations,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )
