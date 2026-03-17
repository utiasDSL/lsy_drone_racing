"""Client-side environment for multi-drone racing with host-client architecture.

The RealMultiDroneRaceEnvClient operates as a client in a host-client system:
- Receives coordination messages from the host via Zenoh
- Manages a single drone's state and control
- Sends state updates to the host for supervision
- Handles local observation and gate tracking
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np
import rclpy
import zenoh
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from gymnasium import Env

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.zenoh_utils import (
    ClientPingMessage,
    ClientStateMessage,
    HostInitializedMessage,
    HostPongMessage,
    HostReadyMessage,
    RaceStartMessage,
    ZenohPublisher,
    ZenohSubscriber,
    compute_latency_ms,
    create_zenoh_session,
    deserialize_message,
)

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ClientEnvData:
    """Auxiliary state for the client-side environment, mirroring :class:`EnvData`."""

    def __init__(self, n_drones: int, n_gates: int, n_obstacles: int):
        self.target_gate = np.zeros(n_drones, dtype=int)
        self.gates_visited = np.zeros((n_drones, n_gates), dtype=bool)
        self.obstacles_visited = np.zeros((n_drones, n_obstacles), dtype=bool)
        self.last_drone_pos = np.zeros((n_drones, 3), dtype=np.float32)
        self.taken_off = False

    def reset(self, last_drone_pos: NDArray[np.float32]):
        """Reset all dynamic fields and seed last drone positions."""
        self.target_gate[:] = 0
        self.gates_visited[:] = False
        self.obstacles_visited[:] = False
        self.last_drone_pos[:] = last_drone_pos
        self.taken_off = False


class RealMultiDroneRaceEnvClient(Env):
    """Client-side Gymnasium environment for multi-drone racing.

    Runs on each drone's computing unit. Receives host coordination messages via Zenoh,
    computes observations and gate tracking locally, and forwards actions to the host
    which relays them to the physical drone.

    Observation space:
        A dictionary containing the state of all drones in the race, mirroring
        :class:`lsy_drone_racing.envs.multi_drone_race.MultiDroneRaceEnv`.

    Action space:
        A single action vector for the drone identified by ``rank``. See
        :class:`~lsy_drone_racing.envs.real_race_host.CrazyflieWorker` for format details.

    Note:
        rclpy must be initialized before creating this environment.
    """

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
        """Initialize the client-side multi-drone environment.

        Args:
            drones: List of all drones in the race, each with ``id``, ``channel``, and
                ``drone_model`` keys.
            rank: Index of this drone among all drones in the race.
            freq: Control frequency in Hz.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            randomizations: Randomization configuration (unused on the client side).
            sensor_range: Distance in metres at which gate/obstacle true poses are revealed.
            control_mode: Either ``"state"`` or ``"attitude"``.
        """
        self.n_drones = len(drones)
        self.rank = rank
        self.freq = freq
        self.sensor_range = sensor_range
        self.control_mode = control_mode
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]

        self.gates, self.obstacles, self.drones_track = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])

        self.device = jax.devices("cpu")[0]
        self._ros_connector_own: ROSConnector | None = None
        self._ros_connector_others: ROSConnector | None = None
        self.data = ClientEnvData(self.n_drones, self.n_gates, self.n_obstacles)

        self._zenoh_session: zenoh.Session | None = None
        self._host_ready_sub: ZenohSubscriber | None = None
        self._host_initialized_sub: ZenohSubscriber | None = None
        self._host_pong_sub: ZenohSubscriber | None = None
        self._race_start_sub: ZenohSubscriber | None = None
        self._client_state_pub: ZenohPublisher | None = None
        self._client_ping_pub: ZenohPublisher | None = None

        self._host_ready_event = threading.Event()
        self._race_started = False
        self._race_start_time = 0.0
        self._should_stop = False
        self._clock_offset = 0.0

    def _init_ros_connectors(self):
        """Open ROS connectors for own drone (estimator) and others (TF), in parallel."""
        def init_own_connector() -> ROSConnector:
            return ROSConnector(
                estimator_names=[self.drone_name],
                cmd_topic=f"/drones/{self.drone_name}/command",
                timeout=10.0,
            )

        def init_others_connector() -> ROSConnector | None:
            other_names = [n for i, n in enumerate(self.drone_names) if i != self.rank]
            if not other_names:
                return None
            return ROSConnector(tf_names=other_names, timeout=10.0)

        logger.info(f"Client {self.rank}: Initializing ROS connectors...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_own = executor.submit(init_own_connector)
            future_others = executor.submit(init_others_connector)
            self._ros_connector_own = future_own.result()
            self._ros_connector_others = future_others.result()
        logger.info(f"Client {self.rank}: ROS connectors initialized")

    def _get_all_drone_states(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read positions, quaternions, velocities, and angular velocities for all drones.

        Own drone state comes from the high-precision estimator; other drones from TF.
        Fields for unreachable drones are filled with NaN.

        Returns:
            Tuple of ``(pos, quat, vel, ang_vel)``, each of shape ``(n_drones, ...)``.
        """
        pos = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        quat = np.full((self.n_drones, 4), np.nan, dtype=np.float32)
        vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        ang_vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        pos[self.rank] = self._ros_connector_own.pos[self.drone_name]
        quat[self.rank] = self._ros_connector_own.quat[self.drone_name]
        vel[self.rank] = self._ros_connector_own.vel[self.drone_name]
        ang_vel[self.rank] = self._ros_connector_own.ang_vel[self.drone_name]
        if self._ros_connector_others is not None:
            for i, name in enumerate(self.drone_names):
                if i != self.rank:
                    pos[i] = self._ros_connector_others.pos[name]
                    quat[i] = self._ros_connector_others.quat[name]
        return pos, quat, vel, ang_vel

    def _init_zenoh(self):
        """Open a Zenoh session and register all host/client publishers and subscribers."""
        self._zenoh_session = create_zenoh_session()

        def on_host_ready(payload: str):
            try:
                msg = deserialize_message(payload, HostReadyMessage)
                self._host_ready_event.set()
                logger.debug(f"Client {self.rank}: Host ready (latency: {compute_latency_ms(msg.timestamp):.2f}ms)")
            except Exception as e:
                logger.error(f"Client {self.rank}: Error processing host ready message: {e}")

        self._host_ready_sub = ZenohSubscriber(
            self._zenoh_session, "lsy_drone_racing/host/ready", on_host_ready
        )

        def on_host_initialized(payload: str):
            try:
                msg = deserialize_message(payload, HostInitializedMessage)
                if msg.drone_rank == self.rank:
                    self._client_ping_pub.publish(
                        ClientPingMessage(drone_rank=self.rank, client_timestamp=time.time())
                    )
            except Exception as e:
                logger.error(f"Client {self.rank}: Error processing host initialized message: {e}")

        self._host_initialized_sub = ZenohSubscriber(
            self._zenoh_session, "lsy_drone_racing/host/initialized", on_host_initialized
        )

        def on_host_pong(payload: str):
            try:
                msg = deserialize_message(payload, HostPongMessage)
                if msg.drone_rank == self.rank:
                    # Approximate clock offset as host_time - client_time; RTT is typically <10ms
                    self._clock_offset = float(msg.host_timestamp) - time.time()
                    logger.info(f"Client {self.rank}: Clock offset: {self._clock_offset * 1000:.2f}ms")
            except Exception as e:
                logger.error(f"Client {self.rank}: Error processing host pong message: {e}")

        self._host_pong_sub = ZenohSubscriber(
            self._zenoh_session, "lsy_drone_racing/host/pong", on_host_pong
        )

        def on_race_start(payload: str):
            try:
                msg = deserialize_message(payload, RaceStartMessage)
                self._race_started = True
                self._race_start_time = time.time() - msg.elapsed_time
            except Exception as e:
                logger.error(f"Client {self.rank}: Error processing race start message: {e}")

        self._race_start_sub = ZenohSubscriber(
            self._zenoh_session, "lsy_drone_racing/host/race_start", on_race_start
        )

        self._client_state_pub = ZenohPublisher(
            self._zenoh_session, f"lsy_drone_racing/client/{self.rank}/state"
        )
        self._client_ping_pub = ZenohPublisher(
            self._zenoh_session, f"lsy_drone_racing/client/{self.rank}/ping"
        )
        logger.info(f"Client {self.rank}: Zenoh communication initialized")

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and wait for the host to signal readiness.

        Sends dummy state messages at 10 Hz in the background so the host can detect
        this client as ready. Blocks until :class:`HostReadyMessage` is received.

        Args:
            seed: Unused in real environments.
            options: Unused in real environments.

        Returns:
            Initial observation and info dictionaries.

        Raises:
            TimeoutError: If the host does not respond within 120 seconds.
        """
        if self._ros_connector_own is None:
            self._init_ros_connectors()
        if self._zenoh_session is None:
            self._init_zenoh()

        current_positions, _, _, _ = self._get_all_drone_states()
        self.data.reset(current_positions)

        logger.info(f"Client {self.rank}: Waiting for host ready message...")
        stop_sending = threading.Event()

        def send_state_messages():
            while not stop_sending.is_set():
                if self.control_mode == "attitude":
                    dummy_action = np.zeros(4, dtype=np.float32)
                else:
                    dummy_action = np.zeros(13, dtype=np.float32)
                    dummy_action[:3] = current_positions[self.rank]
                    dummy_action[2] = 0.3
                self._send_state_update(dummy_action, stopped=False)
                time.sleep(0.1)

        threading.Thread(target=send_state_messages, daemon=True).start()
        if not self._host_ready_event.wait(timeout=120.0):
            stop_sending.set()
            raise TimeoutError(
                f"Client {self.rank}: Timeout waiting for host ready. "
                "Host may not be running or network connection failed."
            )
        stop_sending.set()
        logger.info(f"Client {self.rank}: Environment reset complete")
        return self.obs(), self.info()

    def lock_until_race_start(self, timeout: float = 60.0):
        """Block until the host broadcasts the first :class:`RaceStartMessage`.

        Args:
            timeout: Maximum time in seconds to wait before raising.

        Raises:
            TimeoutError: If the race does not start within ``timeout`` seconds.
        """
        logger.info(f"Client {self.rank}: Waiting for race to start...")
        t_start = time.time()
        while not self._race_started:
            if time.time() - t_start > timeout:
                raise TimeoutError(f"Client {self.rank}: Timeout waiting for race start after {timeout}s.")
        logger.info(f"Client {self.rank}: Race started")

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a control step: update gate tracking, check bounds, and send the action.

        Args:
            action: Control action for this drone.

        Returns:
            Observation, reward (always 0.0), terminated, truncated (always False), info.
        """
        drone_pos, _, _, _ = self._get_all_drone_states()

        dpos = drone_pos[:, None, :2] - self.gates.pos[None, :, :2]
        self.data.gates_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        dpos = drone_pos[:, None, :2] - self.obstacles.pos[None, :, :2]
        self.data.obstacles_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range

        gate_pos = self.gates.pos[self.data.target_gate]
        gate_quat = self.gates.quat[self.data.target_gate]
        with jax.default_device(self.device):
            passed = gate_passed(drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1

        terminated = bool(self.data.target_gate[self.rank] == -1)
        if np.any((self.pos_limit_low > drone_pos[self.rank]) | (drone_pos[self.rank] > self.pos_limit_high)):
            logger.warning(f"Client {self.rank}: Drone exceeded safety bounds")
            terminated = True

        self._send_action_ros(action)
        self._send_state_update(action, terminated)
        if terminated:
            self._should_stop = True

        return self.obs(), 0.0, terminated, False, self.info()

    def _send_state_update(self, action: NDArray, stopped: bool):
        """Publish a :class:`ClientStateMessage` to the host.

        The timestamp is adjusted by the calibrated clock offset so the host can
        measure accurate latency without clock skew.

        Args:
            action: Current control action.
            stopped: Whether this client has finished or crashed.
        """
        elapsed_time = time.time() - self._race_start_time if self._race_started else 0.0
        self._client_state_pub.publish(
            ClientStateMessage(
                drone_rank=self.rank,
                action=action.tolist() if isinstance(action, np.ndarray) else list(action),
                elapsed_time=elapsed_time,
                timestamp=time.time() + self._clock_offset,
                stopped=stopped or self._should_stop,
                next_gate_idx=int(self.data.target_gate[self.rank]),
            )
        )

    def _send_action_ros(self, action: NDArray):
        """Publish the action to ROS so the external estimator can track motor commands.

        Args:
            action: Control action (only used in attitude mode).
        """
        if self.control_mode == "attitude" and self._ros_connector_own is not None:
            self._ros_connector_own.publish_cmd(action)

    def obs(self) -> dict[str, NDArray]:
        """Return the current observation dictionary."""
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(np.float32)
        drone_pos, drone_quat, drone_vel, drone_ang_vel = self._get_all_drone_states()
        return {
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

    def info(self) -> dict:
        """Return the info dictionary."""
        return {"rank": self.rank}

    def close(self):
        """Send a final stop message and close all Zenoh and ROS connections."""
        logger.info(f"Client {self.rank}: Closing environment...")
        if self._client_state_pub:
            self._send_state_update(np.zeros(4 if self.control_mode == "attitude" else 13), stopped=True)
        for sub in [self._host_ready_sub, self._host_initialized_sub, self._host_pong_sub, self._race_start_sub]:
            if sub:
                sub.close()
        for pub in [self._client_state_pub, self._client_ping_pub]:
            if pub:
                pub.close()
        if self._zenoh_session:
            self._zenoh_session.close()
        logger.info(f"Client {self.rank}: Environment closed")
