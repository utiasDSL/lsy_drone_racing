"""Real-world drone racing host for multi-drone coordination.

The RealRaceHost manages the central coordination of multi-drone racing, including:
- Track validation and drone connection in IDLE state
- Synchronization with clients in INITIALIZED state
- Race operation and client supervision in OPERATION state
- Graceful shutdown in STOPPING state

Communication with clients is handled via ROS2.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cflib
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from lsy_race_msgs.msg import ClientState, HostReady, RaceStart  # type: ignore[import-untyped]
from lsy_race_msgs.srv import CalibrateClock  # type: ignore[import-untyped]
from scipy.spatial.transform import RigidTransform as Tr
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.utils import load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray
    from rclpy.publisher import Publisher

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
        start_event: mp.synchronize.Event,
        failure_event: mp.synchronize.Event,
        init_pose: Tr,
        control_mode: str,
        control_freq: float = 50.0,
    ):
        """Initialize the Crazyflie worker.

        Args:
            rank: Index of this drone among all drones in the race.
            drone_id: Crazyflie hardware ID (used to build the radio URI).
            drone_channel: Radio channel to connect on.
            drone_model: Drone model name for loading thrust/PWM parameters.
            ready_event: Set by the object itself once the drone is connected and initialized.
            stop_event: Set by the host to request a shutdown.
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
        self.connection_event = mp.Event()
        self.connection_lost_event = mp.Event()
        self.init_pose = init_pose
        self.control_mode = control_mode.lower()
        self.control_freq = control_freq

        logging.basicConfig(level=logging.INFO, format=f"[Drone {rank}] %(levelname)s: %(message)s")
        logging.getLogger("cflib").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.drone_name = f"cf{drone_id}"
        self.drone: Crazyflie | None = None
        self.params: dict | None = None
        self.last_msg: ClientState | None = None
        self.action_lock = threading.Lock()
        self._comm: RaceCommNode | None = None
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
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )

    def _on_client_state(self, msg: ClientState):
        """Store the latest action from the client state message."""
        with self.action_lock:
            self.last_msg = msg

        latency_ms = (time.time() - msg.timestamp) * 1000

        self.logger.debug(f"Action received (gate={msg.next_gate_idx}, latency={latency_ms:.2f}ms)")

    def _connect_drone(self):
        """Connect to the Crazyflie drone via radio.

        Power-cycles the drone first, then opens the radio link. Raises on connection
        failure, link loss (e.g. "Too many packets lost"), or timeout.

        Raises:
            RuntimeError: If the connection fails or the link is lost before full connection.
            TimeoutError: If the drone does not connect within 10 seconds.
        """
        self.logger.info(f"Connecting to drone {self.drone_id} on channel {self.drone_channel}...")
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))

        try:
            cflib.crtp.init_drivers()
            uri = f"radio://{self.rank}/{self.drone_channel}/2M/E7E7E7E7{self.drone_id:02X}"
            PowerSwitch(uri).stm_power_cycle()
            time.sleep(2)

            connection_failed_event = threading.Event()

            def on_connected(_: str):
                self.connection_event.set()

            def on_connection_failed(uri_failed: str, msg: str):
                self.logger.error(f"Connection failed to {uri_failed}: {msg}")
                connection_failed_event.set()

            def on_connection_lost(uri_lost: str, msg: str):
                if self.connection_event.is_set():
                    self.logger.warning(f"Connection lost to {uri_lost}: {msg}")
                    self.failure_event.set()
                    self.connection_lost_event.set()

            self.drone.fully_connected.add_callback(on_connected)
            self.drone.connection_failed.add_callback(on_connection_failed)
            self.drone.connection_lost.add_callback(on_connection_lost)
            self.drone.open_link(uri)

            start_time = time.time()
            while time.time() - start_time < 10.0:
                if self.stop_event.is_set():
                    # If interrupted externally before connection, just exit without error.
                    return
                if connection_failed_event.is_set():
                    # If the conenction failed callback was triggered,
                    # set shared failure event and exit
                    self.logger.error(f"Connection failed to drone {self.drone_id}")
                    self.failure_event.set()
                    return

                if self.connection_event.is_set():
                    break
                time.sleep(0.05)

            if not self.connection_event.is_set():
                # If we never connected within the timeout, set shared failure event and exit
                self.failure_event.set()
                self.logger.error(
                    f"Timed out waiting for drone {self.drone_id} on channel {self.drone_channel}."
                )
                return

            self.logger.info(f"Connected to {uri}")
        except Exception:
            # If anything goes wrong during connection, set shared failure event to notify the host
            self.logger.error(f"Exception while connecting to drone {self.drone_id}", exc_info=True)
            self.failure_event.set()
            return

    def _init_ros_comm(self):
        """Subscribe to client state messages for this drone via ROS2."""
        try:
            self._comm = RaceCommNode(f"lsy_race_worker_{self.rank}")
            self._sub = self._comm.node.create_subscription(
                ClientState,
                f"lsy_drone_racing/client/drone_{self.rank}/state",
                self._on_client_state,
                10,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS communication: {e}", exc_info=True)
            self.failure_event.set()

    def _init_ros_connector(self):
        """Open a ROS connector for reading this drone's pose from the estimator."""
        self.logger.info(f"Initializing ROS connector for {self.drone_name}...")
        try:
            self._ros_connector = ROSConnector(
                estimator_names=[self.drone_name],
                cmd_topic=f"/drones/{self.drone_name}/command",
                timeout=10.0,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS connector for {self.drone_name}: {e}")
            self.failure_event.set()

    def _control_loop(self):
        """Send actions to the drone at the configured control frequency."""
        with self.action_lock:
            self.last_msg = None  # Clear any stale message received during initialization

        dt = 1.0 / self.control_freq

        self._last_drone_pos_update = time.perf_counter()
        while not self.stop_event.is_set():
            t_start = time.time()

            with self.action_lock:
                if self.last_msg and (t_start - self.last_msg.timestamp) > 10 * dt:
                    self.logger.error(
                        f"No command received for 10 * {dt:.2f}s, handover control to host..."
                    )
                    break
                if self.last_msg and self.last_msg.stopped:
                    self.logger.info(
                        "Received stop signal from client, handover control to host..."
                    )
                    break
                action = list(self.last_msg.action) if self.last_msg else None

            if action is not None:
                action_array = (
                    np.array(action) if isinstance(action, (list, tuple)) else np.array([action])
                )
                self._send_action(action_array)

            if (t := time.perf_counter()) - self._last_drone_pos_update > 1 / self.POS_UPDATE_FREQ:
                pos = self._ros_connector.pos[self.drone_name]
                quat = self._ros_connector.quat[self.drone_name]
                self.drone.extpos.send_extpose(*pos, *quat)
                self._last_drone_pos_update = t
            elapsed = time.time() - t_start

            time.sleep(max(0.0, dt - elapsed))

    def _cleanup(self):
        """Send an emergency stop, close all connections, and shut down ROS."""
        if self._ros_connector:
            self._ros_connector.close()
        if self._comm:
            self._comm.close()
        if self.drone:
            try:
                if self.connection_event.is_set():
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
        """Run the worker: connect to the drone, initialize, and enter the control loop.

        The worker shall never throw any exceptions; any errors must be caught and logged,
        and the failure_event must be set to notify the host.
        """
        rclpy.init()

        def early_stop() -> bool:
            # If the connection lost
            # or any of the other workers reported a failure
            # or we were asked to stop during connection,
            # just exit
            return (
                self.stop_event.is_set()
                or self.failure_event.is_set()
                or self.connection_lost_event.is_set()
            )

        # TODO: Really bad, but I do not know a better way to do it.
        # The motivation is that the Worker should
        # NEITHER throw exceptions NOR receive interrupt signals.
        # Otherwise it would be very chaotic.
        try:
            if early_stop():
                return
            assert self.control_mode in ["attitude", "state"]
            self.params = load_params(physics="first_principles", drone_model=self.drone_model)
            if early_stop():
                return
            self._connect_drone()
            if early_stop():
                # If the connection failed or we were asked to stop during connection,
                # just exit without error
                return
            self._crazyflie_reset()
            if early_stop():
                return
            self._init_ros_connector()
            if early_stop():
                return
            self._init_ros_comm()
            if early_stop():
                return
            self.ready_event.set()
            self.logger.info("Waiting for start signal...")
            while not self.start_event.is_set():
                if early_stop():
                    return
                time.sleep(0.001)
            self._control_loop()
        finally:
            self._cleanup()

    @staticmethod
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

    _initialized: bool = False
    _num_drones: int = 0
    _config: ConfigDict | None = None
    _comm: RaceCommNode | None
    _host_ready_pub: Publisher | None
    _race_start_pub: Publisher | None

    def __init__(self, config: ConfigDict):
        """Initialize the host and open Zenoh communication.

        Args:
            config: Full configuration dictionary (deploy + env sections).
        """
        self._initialized = False
        self._config = config
        self._shutdown_event = threading.Event()
        self._clients_ready: dict[int, bool] = {}
        self._clients_stopped: dict[int, bool] = {}
        self._start_time = time.time()
        self._comm = None
        self._host_ready_pub = None
        self._race_start_pub = None
        self.load_config(config)
        self.init_comm()

    def init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode("lsy_race_host")
        node = self._comm.node
        self._host_ready_pub = node.create_publisher(HostReady, "lsy_drone_racing/host/ready", 10)
        self._race_start_pub = node.create_publisher(
            RaceStart, "lsy_drone_racing/host/race_start", 10
        )
        self._subs = []
        for rank in range(self._num_drones):

            def on_client_state(msg: ClientState, rank: int = rank):
                if not self._clients_ready[rank]:
                    logger.debug(f"Client {rank} ready")
                    self._clients_ready[rank] = True
                if msg.stopped:
                    logger.info(f"Client {rank} stopped (gate={msg.next_gate_idx})")
                    self._clients_stopped[rank] = True

            self._subs.append(
                node.create_subscription(
                    ClientState, f"lsy_drone_racing/client/drone_{rank}/state", on_client_state, 10
                )
            )
        logger.debug("ROS2 communication initialized")

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
        """Release all resources."""
        if self._comm:
            self._comm.close()


class CrazyFlieRealRaceHost(RealRaceHost):
    """Race host implementation for multi-drone racing with Crazyflie drones.

    Each drone runs in its own subprocess (:class:`CrazyflieWorker`) that handles
    radio communication independently. The host coordinates the race lifecycle via
    Zenoh messages to the client processes.
    """

    gates: ConfigDict
    obstacles: ConfigDict
    drones_pose: ConfigDict
    n_gates: int
    pos_limit_high: NDArray[np.float32]
    pos_limit_low: NDArray[np.float32]

    _num_drones: int
    _drone_names: list[str]
    _drone_ids: list[int]
    _drone_channels: list[int]
    _drone_models: list[str]
    _drone_connections: dict[int, DroneConnection] | None
    _drone_control_freq: list[float]
    _drone_control_mode: list[str]
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
        self.gates, self.obstacles, self.drones_pose = load_track(config.env.track)
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
        self._drone_control_mode = [kwargs["control_mode"] for kwargs in config.env.kwargs]
        for rank in range(self._num_drones):
            self._clients_ready[rank] = False
            self._clients_stopped[rank] = False

    def check_track(
        self, rng_config: ConfigDict, check_objects: bool = True, check_drones: bool = True
    ) -> None:
        """Verify that gates, obstacles, and drones are within their allowed tolerances.

        Args:
            rng_config: Randomization config used to determine position tolerances.
            check_objects: Whether to validate gate and obstacle positions.
            check_drones: Whether to validate drone start positions.
        """
        if not check_objects and not check_drones:
            return
        logger.debug("Checking track configuration...")
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
            logger.debug("Track object check passed")
        if check_drones:
            for rank, drone_name in enumerate(self._drone_names):
                check_drone_start_pos(
                    nominal_pos=self.drones_pose.nominal_pos[rank],
                    real_pos=self.drones_pose.pos[rank],
                    rng_config=rng_config,
                    drone_name=drone_name,
                )
            logger.debug("Drone start position check passed")

    def connect_drones(self):
        """Spawn one subprocess per drone and wait until all are connected and ready.

        Raises:
            RuntimeError: If any worker fails to connect or exits prematurely.
            TimeoutError: If all drones are not ready within 10 seconds.
        """
        logger.debug(f"Spawning processes for {self._num_drones} Crazyflie drones...")
        self._drone_connections = {}
        failure_event = self._mp_ctx.Event()

        for rank in range(self._num_drones):
            init_pose = Tr.from_components(
                translation=self.drones_pose.pos[rank],
                rotation=R.from_quat(self.drones_pose.quat[rank]),
            )
            ready_event = self._mp_ctx.Event()
            stop_event = self._mp_ctx.Event()
            process = self._mp_ctx.Process(
                target=CrazyflieWorker.crazyflie_process_worker,
                args=(
                    rank,
                    self._drone_ids[rank],
                    self._drone_channels[rank],
                    self._drone_models[rank],
                    ready_event,
                    stop_event,
                    init_pose,
                    self._drone_control_mode[rank],
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
        while time.time() - start_time < 10.0:
            if all(conn.ready_event.is_set() for conn in self._drone_connections.values()):
                for conn in self._drone_connections.values():
                    conn.connected = True
                    logger.info(f"Drone {conn.rank} ready")
                self._initialized = True
                return

            if failure_event.is_set():
                for other in self._drone_connections.values():
                    other.stop_event.set()
                raise RuntimeError("One or more drone workers failed to connect")

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
        into ``self.gates``, ``self.obstacles``, and ``self.drones_pose``, then closes
        the connectors.

        Args:
            track_obj: Whether to update gate and obstacle poses.
            drones: Whether to update drone start poses.
        """
        if not track_obj and not drones:
            return

        ros_connector: ROSConnector | None = None
        if track_obj:
            try:
                tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
                tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
                ros_connector = ROSConnector(estimator_names=tf_names, timeout=5.0)
                for i in range(self.n_gates):
                    self.gates.pos[i] = ros_connector.pos[f"gate{i + 1}"]
                    self.gates.quat[i] = ros_connector.quat[f"gate{i + 1}"]
                for i in range(self.n_obstacles):
                    self.obstacles.pos[i] = ros_connector.pos[f"obstacle{i + 1}"]
                    self.obstacles.quat[i] = ros_connector.quat[f"obstacle{i + 1}"]
            finally:
                if ros_connector:
                    ros_connector.close()

        if drones:
            ros_connector = None
            try:
                ros_connector = ROSConnector(estimator_names=self._drone_names, timeout=5.0)
                for rank, drone_name in enumerate(self._drone_names):
                    self.drones_pose.pos[rank] = ros_connector.pos[drone_name]
                    self.drones_pose.quat[rank] = ros_connector.quat[drone_name]
            finally:
                if ros_connector:
                    ros_connector.close()

    def close(self):
        """Stop all drone subprocesses and close ROS communication."""
        self._race_start_pub.publish(
            RaceStart(elapsed_time=-1.0, timestamp=time.time(), finished=True)
        )
        if self._drone_connections is not None:
            for conn in self._drone_connections.values():
                if conn.stop_event:
                    conn.stop_event.set()
            for rank, conn in self._drone_connections.items():
                if conn.process and conn.process.is_alive():
                    conn.process.join(timeout=5)
                    if conn.process.is_alive():
                        conn.process.terminate()
                        conn.process.join(timeout=5)
                        if conn.process.is_alive():
                            conn.process.kill()
                            conn.process.join()

        super().close()
        logger.info("Host shutdown complete")

    def _calibrate_client_clocks(self):
        """Expose the clock calibration service and wait for all clients to calibrate.

        Creates a single ``lsy_drone_racing/calibrate_clock`` service server. Clients
        discover it via ``wait_for_service`` and call it N times to estimate the clock
        offset using the midpoint method. The host waits 3 seconds for all clients to
        complete their calls before proceeding.
        """
        logger.info("Starting clock calibration service...")

        def _handler(
            _: CalibrateClock.Request, response: CalibrateClock.Response
        ) -> CalibrateClock.Response:
            response.host_timestamp = time.time()
            return response

        self._calib_srv = self._comm.node.create_service(
            CalibrateClock, "lsy_drone_racing/calibrate_clock", _handler
        )
        time.sleep(1.0)
        logger.info("Clock calibration complete")

    def host_main_loop(self, race_update_freq: float = 50.0):
        """Run the host coordination loop.

        Broadcasts :class:`HostReadyMessage` until all clients signal readiness, then
        performs clock calibration, releases the drone workers, and enters the race loop
        where :class:`RaceStartMessage` is broadcast until all clients report stopping.

        Args:
            race_update_freq: Frequency in Hz at which :class:`RaceStartMessage` is broadcast.

        Raises:
            RuntimeError: If drones have not been connected yet.
            TimeoutError: If clients do not become ready within 300 seconds.
        """
        if not self._initialized:
            raise RuntimeError("Drones must be connected before starting the main loop")

        logger.info("Waiting for all clients...")
        t_start = time.time()
        while time.time() - t_start < 300.0:
            self._host_ready_pub.publish(HostReady(elapsed_time=0.0, timestamp=time.time()))
            if all(self._clients_ready.values()):
                logger.info("All clients ready")
                break
            time.sleep(1.0 / 10)

        if not all(self._clients_ready.values()):
            raise TimeoutError("Timeout waiting for all clients to become ready")

        self._calibrate_client_clocks()
        self._all_clients_ready_event.set()

        logger.info("Race started")
        self._start_time = time.time()

        while True:
            elapsed_time = time.time() - self._start_time
            finished = all(self._clients_stopped.values())
            self._race_start_pub.publish(
                RaceStart(elapsed_time=elapsed_time, timestamp=time.time(), finished=finished)
            )
            if finished:
                logger.info("All clients have stopped")
                break
            time.sleep(1.0 / race_update_freq)
