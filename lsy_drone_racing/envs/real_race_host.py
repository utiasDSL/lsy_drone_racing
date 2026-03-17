"""Real-world drone racing host for multi-drone coordination.

The RealRaceHost manages the central coordination of multi-drone racing, including:
- Track validation and drone connection in IDLE state
- Synchronization with clients in INITIALIZED state
- Race operation and client supervision in OPERATION state
- Graceful shutdown in STOPPING state

Communication with clients is handled via Zenoh pub/sub.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import signal
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cflib
import numpy as np
import rclpy
import zenoh
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from scipy.spatial.transform import Rotation as R, RigidTransform as Tr

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.zenoh_utils import (
    ClientStateMessage,
    HostInitializedMessage,
    HostPongMessage,
    HostReadyMessage,
    RaceStartMessage,
    ZenohPublisher,
    ZenohSubscriber,
    create_zenoh_session,
    deserialize_message,
)

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CrazyflieWorker:
    """Manages a single Crazyflie drone in a dedicated subprocess.

    Connects to the drone via radio, resets it to its initial state, and runs
    a control loop that forwards actions received from the client over Zenoh.
    External position updates from the ROS estimator are forwarded to the drone's
    Kalman filter at a fixed rate.
    """

    POS_UPDATE_FREQ = 30  # Hz at which MoCap poses are forwarded to the drone's Kalman filter

    def __init__(
        self,
        rank: int,
        drone_id: int,
        drone_channel: int,
        drone_model: str,
        ready_event: mp.synchronize.Event,
        stop_event: mp.synchronize.Event,
        init_pose: Tr,
        control_mode: str,
        control_freq: float = 50.0,
        start_event: mp.synchronize.Event | None = None,
        failure_event: mp.synchronize.Event | None = None,
    ):
        """Initialize the Crazyflie worker.

        Args:
            rank: Index of this drone among all drones in the race.
            drone_id: Crazyflie hardware ID (used to build the radio URI).
            drone_channel: Radio channel to connect on.
            drone_model: Drone model name for loading thrust/PWM parameters.
            ready_event: Set once the drone is connected and initialized.
            stop_event: Set by the host to request a graceful shutdown.
            init_pose: Initial pose used to seed the drone's Kalman filter.
            control_mode: Either ``"attitude"`` or ``"state"``.
            control_freq: Frequency in Hz at which actions are forwarded to the drone.
            start_event: Set by the host once all clients are ready; the control loop
                blocks until this is set so that all drones start simultaneously.
            failure_event: Shared event set on connection failure to notify all other
                workers and the host.
        """
        self.rank = rank
        self.drone_id = drone_id
        self.drone_channel = drone_channel
        self.drone_model = drone_model
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.start_event = start_event
        self.failure_event = failure_event
        self.init_pose = init_pose
        self.control_mode = control_mode.lower()
        self.control_freq = control_freq

        logging.basicConfig(level=logging.INFO, format=f"[Drone {rank}] %(levelname)s: %(message)s")
        logging.getLogger("cflib").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.drone_name = f"cf{drone_id}"
        self.drone: Crazyflie | None = None
        self.zenoh_session: zenoh.Session | None = None
        self.state_sub: ZenohSubscriber | None = None
        self.params: dict | None = None
        self.last_action: list | None = None
        self.action_lock = threading.Lock()
        self._ros_connector: ROSConnector | None = None
        self._last_drone_pos_update: float = 0.0

    def _apply_drone_settings(self):
        """Apply firmware settings required for racing.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        self.drone.param.set_value("stabilizer.estimator", 2)  # 1: complementary, 2: kalman
        time.sleep(0.1)
        self.drone.param.set_value("supervisor.tmblChckEn", 1)
        self.drone.param.set_value("stabilizer.controller", 2)  # 1: PID, 2: Mellinger
        self.drone.param.set_value("flightmode.stabModeRoll", 1)  # 0: rate, 1: angle
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)

    def _crazyflie_reset(self):
        """Arm the drone and reset the Kalman filter to the initial pose."""
        self.drone.platform.send_arming_request(True)
        self._apply_drone_settings()
        pos = self.init_pose.translation
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        yaw = self.init_pose.rotation.as_euler("xyz", degrees=False)[2]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")
        if self.control_mode == "attitude":
            # Required to unlock the firmware's thrust protection before the first setpoint
            self.drone.commander.send_setpoint(0, 0, 0, 0)

    def _send_action(self, action: NDArray[np.float32]):
        """Forward an action to the drone.

        Args:
            action: For attitude mode, a 4-element array ``[roll, pitch, yaw, thrust]`` in
                radians and Newtons. For state mode, a 13-element array
                ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rollrate, pitchrate, yawrate]``.
        """
        if self.control_mode == "attitude":
            if action.shape[0] != 4:
                raise ValueError(f"Attitude action must have shape (4,), got {action.shape}")
            pwm = force2pwm(action[3], self.params["thrust_max"] * 4, self.params["pwm_max"])
            pwm = np.clip(pwm, self.params["pwm_min"], self.params["pwm_max"])
            self.drone.commander.send_setpoint(*np.rad2deg(action[:3]), int(pwm))
        else:
            if action.shape[0] != 13:
                raise ValueError(f"State action must have shape (13,), got {action.shape}")
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(pos, vel, acc, quat, rollrate, pitchrate, yawrate)

    def _on_client_state(self, payload: str):
        """Store the latest action from the client state message."""
        msg = deserialize_message(payload, ClientStateMessage)
        with self.action_lock:
            self.last_action = msg.action
        latency_ms = (time.time() - msg.timestamp) * 1000
        self.logger.debug(f"Action received (gate={msg.next_gate_idx}, latency={latency_ms:.2f}ms)")

    def _connect_drone(self):
        """Connect to the Crazyflie drone via radio.

        Power-cycles the drone first, then opens the radio link. Raises on connection
        failure, link loss (e.g. "Too many packets lost"), or timeout.

        Raises:
            InterruptedError: If ``stop_event`` is set during the connection attempt.
            RuntimeError: If the connection fails or the link is lost before full connection.
            TimeoutError: If the drone does not connect within 10 seconds.
        """
        self.logger.info(f"Connecting to drone {self.drone_id} on channel {self.drone_channel}...")
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        cflib.crtp.init_drivers()
        uri = f"radio://{self.rank}/{self.drone_channel}/2M/E7E7E7E7{self.drone_id:02X}"
        failure_msg: dict[str, str] = {"message": ""}
        try:
            PowerSwitch(uri).stm_power_cycle()
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if self.stop_event.is_set():
                    raise InterruptedError("Stop requested during power-cycle wait")
                time.sleep(0.05)
        except InterruptedError:
            raise
        except Exception as e:
            self.failure_event.set()
            raise e

        connection_event = mp.Event()
        connection_failed_event = mp.Event()

        def on_connected(_: str):
            connection_event.set()

        def on_connection_failed(uri_failed: str, msg: str):
            failure_msg["message"] = f"Connection failed to {uri_failed}: {msg}"
            connection_failed_event.set()

        def on_connection_lost(uri_lost: str, msg: str):
            if not connection_event.is_set():
                failure_msg["message"] = f"Connection lost to {uri_lost}: {msg}"
                connection_failed_event.set()

        def on_disconnected(uri_disconnected: str):
            self.logger.info(f"Disconnected from {uri_disconnected}")

        self.drone.fully_connected.add_callback(on_connected)
        self.drone.connection_failed.add_callback(on_connection_failed)
        self.drone.connection_lost.add_callback(on_connection_lost)
        self.drone.disconnected.add_callback(on_disconnected)
        self.drone.open_link(uri)

        start_time = time.time()
        while time.time() - start_time < 10.0:
            if self.stop_event.is_set():
                raise InterruptedError("Stop requested while connecting to drone")
            if connection_failed_event.is_set():
                self.failure_event.set()
                raise RuntimeError(failure_msg["message"])
            if connection_event.is_set():
                break
            time.sleep(0.05)

        if not connection_event.is_set():
            self.failure_event.set()
            raise TimeoutError(f"Timed out waiting for drone {self.drone_id} on channel {self.drone_channel}.")

        self.logger.info(f"Connected to {uri}")

    def _init_zenoh(self):
        """Open a Zenoh session and subscribe to client state messages for this drone."""
        self.zenoh_session = create_zenoh_session()
        self.state_sub = ZenohSubscriber(
            self.zenoh_session,
            f"lsy_drone_racing/client/{self.rank}/state",
            self._on_client_state,
        )

    def _init_ros_connector(self):
        """Open a ROS connector for reading this drone's pose from the estimator."""
        self.logger.info(f"Initializing ROS connector for {self.drone_name}...")
        self._ros_connector = ROSConnector(
            estimator_names=[self.drone_name],
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )

    def _control_loop(self):
        """Send actions to the drone at the configured control frequency.

        Blocks until ``start_event`` is set, then forwards the latest action from
        the client to the drone and sends periodic external position updates to the
        drone's Kalman filter.
        """
        dt = 1.0 / self.control_freq
        self.logger.info("Waiting for start signal...")
        while not self.start_event.is_set():
            if self.stop_event.is_set():
                return
            time.sleep(0.001)

        self._last_drone_pos_update = time.perf_counter()
        while not self.stop_event.is_set():
            t_start = time.time()
            with self.action_lock:
                action = self.last_action
            if action is not None:
                action_array = np.array(action) if isinstance(action, (list, tuple)) else np.array([action])
                self._send_action(action_array)
            elapsed = time.time() - t_start
            if elapsed > dt:
                self.logger.warning(f"Control loop overrun: {elapsed*1000:.1f}ms (budget: {dt*1000:.1f}ms)")
            if (t := time.perf_counter()) - self._last_drone_pos_update > 1 / self.POS_UPDATE_FREQ:
                pos = self._ros_connector.pos[self.drone_name]
                quat = self._ros_connector.quat[self.drone_name]
                self.drone.extpos.send_extpose(*pos, *quat)
                self._last_drone_pos_update = t
            time.sleep(max(0.0, dt - elapsed))

    def _cleanup(self):
        """Send an emergency stop, close all connections, and shut down ROS."""
        if self._ros_connector:
            self._ros_connector.close()
        if self.state_sub:
            self.state_sub.close()
        if self.zenoh_session:
            self.zenoh_session.close()
        if self.drone:
            try:
                pk = CRTPPacket()
                pk.port = CRTPPort.LOCALIZATION
                pk.channel = Localization.GENERIC_CH
                pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
                self.drone.send_packet(pk)
            finally:
                self.drone.close_link()
        rclpy.shutdown()
        self.logger.info("Drone process finished")

    def run(self):
        """Run the worker: connect to the drone, initialize, and enter the control loop."""
        rclpy.init()
        try:
            if self.stop_event.is_set():
                return
            assert self.control_mode in ["attitude", "state"]
            self.params = load_params(physics="first_principles", drone_model=self.drone_model)
            if self.stop_event.is_set():
                return
            try:
                self._connect_drone()
            except InterruptedError:
                return
            if self.stop_event.is_set():
                return
            self._crazyflie_reset()
            if self.stop_event.is_set():
                return
            self._init_ros_connector()
            self._init_zenoh()
            self.ready_event.set()
            if self.stop_event.is_set():
                return
            self._control_loop()
        finally:
            self._cleanup()


def crazyflie_process_worker(
    rank: int,
    drone_id: int,
    drone_channel: int,
    drone_model: str,
    ready_event: mp.synchronize.Event,
    stop_event: mp.synchronize.Event,
    init_pose: Tr,
    control_mode: str,
    start_event: mp.synchronize.Event,
    failure_event: mp.synchronize.Event | None = None,
    control_freq: float = 50.0,
):
    """Multiprocessing entry point that creates and runs a :class:`CrazyflieWorker`.

    SIGINT is ignored so that only the host process handles keyboard interrupts.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    CrazyflieWorker(
        rank=rank,
        drone_id=drone_id,
        drone_channel=drone_channel,
        drone_model=drone_model,
        ready_event=ready_event,
        stop_event=stop_event,
        init_pose=init_pose,
        control_mode=control_mode,
        control_freq=control_freq,
        start_event=start_event,
        failure_event=failure_event,
    ).run()


class RealRaceHostState(Enum):
    """State machine states for the :class:`RealRaceHost`.

    Attributes:
        IDLE: Loading track configuration and checking track/drone positions.
        INITIALIZED: Connected to all drones, waiting for clients to signal readiness.
        OPERATION: Race in progress, forwarding actions and supervising clients.
        STOPPING: All clients finished, shutting down gracefully.
    """

    IDLE = 0
    INITIALIZED = 1
    OPERATION = 2
    STOPPING = 3


@dataclass
class DroneConnection:
    """Book-keeping for a single drone subprocess managed by the host."""

    rank: int
    drone_id: int
    drone_channel: int
    process: mp.Process | None = None
    ready_event: mp.synchronize.Event | None = None
    stop_event: mp.synchronize.Event | None = None
    failure_event: mp.synchronize.Event | None = None
    connected: bool = False


class RealRaceHost:
    """Base class for multi-drone race hosts.

    Subclasses implement :meth:`load_config`, :meth:`connect_drones`,
    :meth:`host_main_loop`, and :meth:`close` for a specific drone platform.
    """

    state: RealRaceHostState
    _num_drones: int = 0
    _config: ConfigDict | None = None
    _zenoh_session: zenoh.Session | None
    _host_ready_pub: ZenohPublisher | None
    _race_start_pub: ZenohPublisher | None
    _client_state_subs: dict[int, ZenohSubscriber] | None

    def __init__(self, config: ConfigDict):
        """Initialize the host and open Zenoh communication.

        Args:
            config: Full configuration dictionary (deploy + env sections).
        """
        self.state = RealRaceHostState.IDLE
        self._config = config
        self._shutdown_event = threading.Event()
        self._clients_ready: dict[int, bool] = {}
        self._clients_stopped: dict[int, bool] = {}
        self._start_time = time.time()
        self._host_ready_pub = None
        self._race_start_pub = None
        self._client_state_subs = None
        self.load_config(config)
        self.init_zenoh()

    def init_zenoh(self, conf: zenoh.Config | None = None) -> zenoh.Session:
        """Open a Zenoh session and set up all publishers and subscribers.

        Args:
            conf: Optional Zenoh configuration. Uses defaults if ``None``.

        Returns:
            The opened Zenoh session.
        """
        self._client_state_subs = {}
        self._zenoh_session = create_zenoh_session(conf)
        self._host_ready_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/ready")
        self._host_initialized_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/initialized")
        self._host_pong_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/pong")
        self._race_start_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/race_start")

        for rank in range(self._num_drones):
            def on_client_state(payload: str, rank: int = rank):
                msg = deserialize_message(payload, ClientStateMessage)
                if not self._clients_ready[rank]:
                    logger.debug(f"Client {rank} ready")
                    self._clients_ready[rank] = True
                if msg.stopped:
                    logger.info(f"Client {rank} stopped (gate={msg.next_gate_idx})")
                    self._clients_stopped[rank] = True

            self._client_state_subs[rank] = ZenohSubscriber(
                self._zenoh_session,
                f"lsy_drone_racing/client/{rank}/state",
                on_client_state,
            )

        self._client_ping_subs: dict[int, ZenohSubscriber] = {}
        for rank in range(self._num_drones):
            def on_client_ping(payload: str, rank: int = rank):
                self._host_pong_pub.publish(HostPongMessage(drone_rank=rank, host_timestamp=time.time()))

            self._client_ping_subs[rank] = ZenohSubscriber(
                self._zenoh_session,
                f"lsy_drone_racing/client/{rank}/ping",
                on_client_ping,
            )

        logger.info("Zenoh communication initialized")
        return self._zenoh_session

    def load_config(self, config: ConfigDict):
        """Load and validate the configuration. Must be implemented by subclasses."""
        raise NotImplementedError

    def connect_drones(self):
        """Connect to all drones. Must be implemented by subclasses."""
        raise NotImplementedError

    def host_main_loop(self):
        """Run the host's main coordination loop. Must be implemented by subclasses."""
        raise NotImplementedError

    def close(self):
        """Release all resources. Must be implemented by subclasses."""
        raise NotImplementedError


class CrazyFlieRealRaceHost(RealRaceHost):
    """Race host implementation for multi-drone racing with Crazyflie drones.

    Each drone runs in its own subprocess (:class:`CrazyflieWorker`) that handles
    radio communication independently. The host coordinates the race lifecycle via
    Zenoh messages to the client processes.
    """

    _drone_names: list[str]
    _drone_ids: list[int]
    _drone_channels: list[int]
    _drone_models: list[str]
    _drone_connections: dict[int, DroneConnection] | None
    _drone_control_freq: list[float]
    _all_clients_ready_event: mp.synchronize.Event | None
    _mp_ctx: mp.context.BaseContext

    def __init__(self, config: ConfigDict):
        """Initialize the host.

        Args:
            config: Full configuration dictionary (deploy + env sections).
        """
        super().__init__(config)
        self._mp_ctx = mp.get_context("spawn")
        self._all_clients_ready_event = self._mp_ctx.Event()
        self._drone_connections = None

    def load_config(self, config: ConfigDict):
        """Parse drone and track information from the configuration.

        Args:
            config: Full configuration dictionary (deploy + env sections).
        """
        self._config = config
        self.gates, self.obstacles, self.drones_track = load_track(config.env.track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(config.env.track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(config.env.track.safety_limits["pos_limit_high"])
        self._num_drones = len(config.deploy.drones)
        self._drone_names = [f"cf{drone['id']}" for drone in config.deploy.drones]
        self._drone_ids = [drone["id"] for drone in config.deploy.drones]
        self._drone_channels = [drone["channel"] for drone in config.deploy.drones]
        self._drone_models = [drone["drone_model"] for drone in config.deploy.drones]
        self._drone_control_freq = [kwargs["freq"] for kwargs in config.env.kwargs]
        for rank in range(self._num_drones):
            self._clients_ready[rank] = False
            self._clients_stopped[rank] = False

    def check_track(
        self,
        rng_config: ConfigDict,
        check_objects: bool = True,
        check_drones: bool = True,
    ) -> None:
        """Verify that gates, obstacles, and drones are within their allowed tolerances.

        Args:
            rng_config: Randomization config used to determine position tolerances.
            check_objects: Whether to validate gate and obstacle positions.
            check_drones: Whether to validate drone start positions.
        """
        if not check_objects and not check_drones:
            return
        logger.info("Checking track configuration...")
        if check_objects:
            check_race_track(
                gates_pos=self.gates.pos,
                nominal_gates_pos=self.gates.nominal_pos,
                gates_quat=self.gates.quat,
                nominal_gates_quat=self.gates.nominal_quat,
                obstacles_pos=self.obstacles.pos,
                nominal_obstacles_pos=self.obstacles.nominal_pos,
                rng_config=rng_config,
            )
            logger.info("Track object check passed")
        if check_drones:
            for rank, drone_name in enumerate(self._drone_names):
                check_drone_start_pos(
                    nominal_pos=self.drones_track.nominal_pos[rank],
                    real_pos=self.drones_track.pos[rank],
                    rng_config=rng_config,
                    drone_name=drone_name,
                )
            logger.info("Drone start position check passed")

    def connect_drones(self):
        """Spawn one subprocess per drone and wait until all are connected and ready.

        Raises:
            RuntimeError: If any worker fails to connect or exits prematurely.
            TimeoutError: If all drones are not ready within 100 seconds.
        """
        logger.info(f"Spawning processes for {self._num_drones} Crazyflie drones...")
        self._drone_connections = {}
        failure_event = self._mp_ctx.Event()

        for rank in range(self._num_drones):
            init_pose = Tr.from_components(
                translation=self.drones_track.pos[rank],
                rotation=R.from_quat(self.drones_track.quat[rank]),
            )
            ready_event = self._mp_ctx.Event()
            stop_event = self._mp_ctx.Event()
            process = self._mp_ctx.Process(
                target=crazyflie_process_worker,
                args=(
                    rank,
                    self._drone_ids[rank],
                    self._drone_channels[rank],
                    self._drone_models[rank],
                    ready_event,
                    stop_event,
                    init_pose,
                    self._config.env.kwargs[rank]["control_mode"],
                    self._all_clients_ready_event,
                    failure_event,
                    self._drone_control_freq[rank],
                ),
                name=f"CrazyflieProcess-{rank}",
            )
            process.start()
            self._drone_connections[rank] = DroneConnection(
                rank=rank,
                drone_id=self._drone_ids[rank],
                drone_channel=self._drone_channels[rank],
                process=process,
                ready_event=ready_event,
                stop_event=stop_event,
                failure_event=failure_event,
            )
            logger.debug(f"Spawned process for drone {rank} (PID: {process.pid})")

        start_time = time.time()
        while time.time() - start_time < 100.0:
            if failure_event.is_set():
                for conn in self._drone_connections.values():
                    conn.stop_event.set()
                raise RuntimeError("A Crazyflie worker failed to connect. Stopping all workers.")

            all_ready = all(conn.ready_event.is_set() for conn in self._drone_connections.values())
            if all_ready:
                for conn in self._drone_connections.values():
                    conn.connected = True
                    logger.info(f"Drone {conn.rank} ready")
                self.state = RealRaceHostState.INITIALIZED
                return

            for rank, conn in self._drone_connections.items():
                if conn.process and not conn.process.is_alive() and not conn.ready_event.is_set():
                    for other in self._drone_connections.values():
                        other.stop_event.set()
                    raise RuntimeError(f"Process for drone {rank} terminated unexpectedly")

            time.sleep(0.1)

        for conn in self._drone_connections.values():
            conn.stop_event.set()
        raise TimeoutError(f"Timeout waiting for {self._num_drones} Crazyflie processes to connect")

    def update_poses(self, track_obj: bool = False, drones: bool = False) -> None:
        """Update gate, obstacle, and/or drone poses from the motion capture system.

        Initializes temporary ROS connectors, reads the current TF poses, writes them
        into ``self.gates``, ``self.obstacles``, and ``self.drones_track``, then closes
        the connectors.

        Args:
            track_obj: Whether to update gate and obstacle poses.
            drones: Whether to update drone start poses.
        """
        if not track_obj and not drones:
            return

        logger.info("Reading poses from motion capture system...")
        ros_connector_track: ROSConnector | None = None
        ros_connector_drones: ROSConnector | None = None

        def init_track_connector() -> ROSConnector:
            tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
            tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
            return ROSConnector(tf_names=tf_names, timeout=10.0)

        def init_drone_connector() -> ROSConnector:
            return ROSConnector(tf_names=self._drone_names, timeout=10.0)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_track = executor.submit(init_track_connector) if track_obj else None
                future_drones = executor.submit(init_drone_connector) if drones else None
                if future_track:
                    ros_connector_track = future_track.result()
                if future_drones:
                    ros_connector_drones = future_drones.result()

            if track_obj:
                for i in range(self.n_gates):
                    self.gates.pos[i] = ros_connector_track.pos[f"gate{i + 1}"]
                    self.gates.quat[i] = ros_connector_track.quat[f"gate{i + 1}"]
                for i in range(self.n_obstacles):
                    self.obstacles.pos[i] = ros_connector_track.pos[f"obstacle{i + 1}"]
                    self.obstacles.quat[i] = ros_connector_track.quat[f"obstacle{i + 1}"]
            if drones:
                for rank, drone_name in enumerate(self._drone_names):
                    self.drones_track.pos[rank] = ros_connector_drones.pos[drone_name]
                    self.drones_track.quat[rank] = ros_connector_drones.quat[drone_name]
        finally:
            if ros_connector_track:
                ros_connector_track.close()
            if ros_connector_drones:
                ros_connector_drones.close()

    def close(self):
        """Stop all drone subprocesses and close Zenoh communication."""
        logger.info("Host shutting down...")
        if self._drone_connections is not None:
            for conn in self._drone_connections.values():
                if conn.stop_event:
                    conn.stop_event.set()
            for rank, conn in self._drone_connections.items():
                if conn.process and conn.process.is_alive():
                    conn.process.join(timeout=5)
                    if conn.process.is_alive():
                        logger.warning(f"Process {rank} unresponsive, terminating...")
                        conn.process.terminate()
                        conn.process.join(timeout=2)
                        if conn.process.is_alive():
                            conn.process.kill()
                            conn.process.join()

        for pub in [self._host_ready_pub, self._host_initialized_pub, self._host_pong_pub, self._race_start_pub]:
            if pub:
                pub.close()
        if self._client_state_subs:
            for sub in self._client_state_subs.values():
                sub.close()
        if self._client_ping_subs:
            for sub in self._client_ping_subs.values():
                sub.close()
        if self._zenoh_session:
            self._zenoh_session.close()
        logger.info("Host shutdown complete")

    def _calibrate_client_clocks(self):
        """Trigger ping-pong clock calibration with all clients.

        Broadcasts :class:`HostInitializedMessage` to all clients, prompting each to
        send a ping. The host immediately responds with a pong containing its timestamp,
        allowing clients to estimate the clock offset. Waits 3 seconds for calibration
        to complete before returning.
        """
        logger.info("Triggering client clock calibration...")
        for rank in range(self._num_drones):
            self._host_initialized_pub.publish(HostInitializedMessage(drone_rank=rank, timestamp=time.time()))
        time.sleep(3.0)
        logger.info("Clock calibration complete")

    def host_main_loop(self, race_update_freq: float = 50.0):
        """Run the host coordination loop.

        Broadcasts :class:`HostReadyMessage` until all clients signal readiness, then
        performs clock calibration, releases the drone workers, and enters the race loop
        where :class:`RaceStartMessage` is broadcast until all clients report stopping.

        Args:
            race_update_freq: Frequency in Hz at which :class:`RaceStartMessage` is broadcast.

        Raises:
            RuntimeError: If the host is not in :attr:`RealRaceHostState.INITIALIZED` state.
            TimeoutError: If clients do not become ready within 300 seconds.
        """
        if self.state != RealRaceHostState.INITIALIZED:
            raise RuntimeError("Host must be in INITIALIZED state before starting the main loop")

        logger.info("Waiting for all clients...")
        t_start = time.time()
        while time.time() - t_start < 300.0:
            self._host_ready_pub.publish(HostReadyMessage(elapsed_time=0.0, timestamp=time.time()))
            if all(self._clients_ready.values()):
                logger.info("All clients ready")
                break
            time.sleep(1.0 / 10)

        if not all(self._clients_ready.values()):
            raise TimeoutError("Timeout waiting for all clients to become ready")

        self._calibrate_client_clocks()
        self._all_clients_ready_event.set()

        logger.info("Race started")
        self.state = RealRaceHostState.OPERATION
        self._start_time = time.time()

        while self.state == RealRaceHostState.OPERATION:
            elapsed_time = time.time() - self._start_time
            finished = all(self._clients_stopped.values())
            self._race_start_pub.publish(
                RaceStartMessage(elapsed_time=elapsed_time, timestamp=time.time(), finished=finished)
            )
            if finished:
                logger.info("All clients stopped")
                self.state = RealRaceHostState.STOPPING
                break
            time.sleep(1.0 / race_update_freq)
