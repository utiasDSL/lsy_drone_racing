"""Client-side environment for multi-drone racing with host-client architecture.

The RealMultiDroneRaceEnvClient operates as a client in a host-client system:
- Receives coordination messages from the host via ROS2
- Manages a single drone's state and control
- Sends control actions and state updates to the host for supervision
- Handles local observation and gate tracking
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from gymnasium import Env
from lsy_race_msgs.msg import ClientState, HostReady, RaceStart  # type: ignore[import-untyped]
from lsy_race_msgs.srv import CalibrateClock  # type: ignore[import-untyped]

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode, calibrate_clock, compute_latency_ms
from lsy_drone_racing.utils.ros import track_poses

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ClientEnvData:
    """Auxiliary state for the client-side environment, mirroring :class:`EnvData`."""

    def __init__(self, n_drones: int, n_gates: int, n_obstacles: int):
        """Initialize all dynamic fields to default values."""
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

    Runs on each drone's computing unit. Receives host coordination messages via ROS2,
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

        self._comm: RaceCommNode | None = None
        self._client_state_pub: Any = None
        self._clock_calib_client: Any = None

        self._host_ready_event = threading.Event()
        self._race_started = False
        self._race_start_time = 0.0
        self._clock_offset = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and wait for the host to signal readiness.

        Args:
            seed: Unused in real environments.
            options: Unused in real environments.

        Returns:
            Initial observation and info dictionaries.

        Raises:
            TimeoutError: If the host does not respond within 120 seconds.
        """
        if options and options.get("real_track_objects", False):
            self.gates.pos, self.gates.quat, self.obstacles.pos = track_poses(
                self.n_gates, self.n_obstacles
            )

        if self._ros_connector_own is None:
            self._init_ros_connectors()
        if self._comm is None:
            self._init_comm()

        current_pos, _, _, _ = self._get_all_drone_states()
        self.data.reset(current_pos)

        logger.debug(f"Environment reset complete")
        return self.obs(), self.info()

    def lock_until_race_start(self, timeout: float = 60.0):
        """Sends dummy state messages at the control frequency (``self.freq`` Hz) in the
        background so the host can detect this client as ready. Blocks until
        the race starts.

        Calibrate the clock offset and block until the host broadcasts :class:`RaceStartMessage`.

        Calls the host's calibration service (blocks until available), estimates the
        clock offset via N round-trips, then waits for the race start signal.

        Args:
            timeout: Maximum time in seconds to wait for calibration and race start.

        Raises:
            TimeoutError: If calibration or race start exceeds ``timeout`` seconds.
        """
        logger.info(f"Waiting for host ready message...")
        stop_sending = threading.Event()

        def send_state_messages():
            while not stop_sending.is_set():
                if self.control_mode == "attitude":
                    dummy_action = np.zeros(4, dtype=np.float32)
                else:
                    dummy_action = np.zeros(13, dtype=np.float32)
                    dummy_action[:3] = self._ros_connector_own.pos[self.drone_name]
                self._send_state_update(dummy_action, stopped=False)
                time.sleep(1 / self.freq)

        threading.Thread(target=send_state_messages, daemon=True).start()

        if not self._host_ready_event.wait(timeout=timeout):
            stop_sending.set()
            raise TimeoutError(
                f"Timeout waiting for host ready. "
                "Host may not be running or network connection failed."
            )

        logger.info(f"Received host ready message.")
        self._clock_offset = calibrate_clock(self._clock_calib_client, n=5, timeout=timeout)
        logger.info(f"Clock offset = {self._clock_offset * 1000:.2f}ms")
        logger.info(f"Waiting for race start")

        t_start = time.time()
        while not self._race_started:
            if time.time() - t_start > timeout:
                raise TimeoutError(f"Timeout waiting for race start after {timeout}s.")
            time.sleep(0.001)
        stop_sending.set()
        logger.info(f"Race starts!")

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
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45)
            )
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1

        terminated = bool(self.data.target_gate[self.rank] == -1)
        if np.any(
            (self.pos_limit_low > drone_pos[self.rank])
            | (drone_pos[self.rank] > self.pos_limit_high)
        ):
            logger.warning(f"Drone exceeded safety bounds")
            terminated = True

        if self.control_mode == "attitude" and self._ros_connector_own:
            self._ros_connector_own.publish_cmd(action)

        self._send_state_update(action, terminated)

        return self.obs(), 0.0, terminated, False, self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the current observation dictionary."""
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
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
        """Send a final stop message and close all ROS connections."""
        logger.info(f"Closing environment...")
        if self._client_state_pub:
            try:
                self._send_state_update(
                    np.zeros(4 if self.control_mode == "attitude" else 13), stopped=True
                )
                time.sleep(0.1)  # allow the executor thread to flush the message before shutdown
            except Exception as e:
                logger.warning(f"Could not send final stop message: {e}")
        if self._comm:
            self._comm.close()
        if self._ros_connector_own:
            self._ros_connector_own.close()
        if self._ros_connector_others:
            self._ros_connector_others.close()
        logger.debug(f"Environment closed")

    def _send_state_update(self, action: NDArray, stopped: bool):
        """Publish a :class:`ClientStateMessage` to the host.

        The timestamp is adjusted by the calibrated clock offset so the host can
        measure accurate latency without clock skew.

        Args:
            action: Current control action.
            stopped: Whether this client has finished or crashed.
        """
        elapsed_time = time.time() - self._race_start_time if self._race_started else 0.0
        msg = ClientState()
        msg.drone_rank = self.rank
        msg.action = action.tolist() if isinstance(action, np.ndarray) else list(action)
        msg.elapsed_time = elapsed_time
        msg.timestamp = time.time() + self._clock_offset
        msg.stopped = stopped
        msg.next_gate_idx = int(self.data.target_gate[self.rank])
        self._client_state_pub.publish(msg)

    def _init_ros_connectors(self):
        """Open ROS connectors for own drone (estimator) and others (TF)."""
        self._ros_connector_own = ROSConnector(
            estimator_names=[self.drone_name],
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )

        other_names = [n for i, n in enumerate(self.drone_names) if i != self.rank]
        if other_names:
            self._ros_connector_others = ROSConnector(tf_names=other_names, timeout=10.0)

    def _init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode(f"lsy_race_client_{self.rank}")
        node = self._comm.node

        def on_host_ready(msg: HostReady):
            self._host_ready_event.set()
            logger.debug(f"Host ready (latency: {compute_latency_ms(msg.timestamp):.2f}ms)")

        def on_race_start(msg: RaceStart):
            self._race_started = True
            self._race_start_time = time.time() - msg.elapsed_time
            self._host_terminate = bool(msg.finished)

        self._subs = [
            node.create_subscription(HostReady, "lsy_drone_racing/host/ready", on_host_ready, 10),
            node.create_subscription(
                RaceStart, "lsy_drone_racing/host/race_start", on_race_start, 10
            ),
        ]
        self._client_state_pub = node.create_publisher(
            ClientState, f"lsy_drone_racing/client/drone_{self.rank}/state", 10
        )
        self._clock_calib_client = node.create_client(
            CalibrateClock, "lsy_drone_racing/calibrate_clock"
        )
        logger.debug(f"ROS2 communication initialized")

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
                    pos[i] = self._ros_connector_others.pos.get(name, np.nan)
                    quat[i] = self._ros_connector_others.quat.get(name, np.nan)
                    # vel[i] = self._ros_connector_others.vel.get(name, np.nan)
                    # ang_vel[i] = self._ros_connector_others.ang_vel.get(name, np.nan)
        return pos, quat, vel, ang_vel
