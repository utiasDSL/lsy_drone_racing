"""Real-world drone racing host for multi-drone coordination.

The RealRaceHost manages the central coordination of multi-drone racing, including:
- Track validation and drone connection in IDLE state
- Synchronization with clients in INITIALIZED state
- Race operation and client supervision in OPERATION state
- Graceful shutdown in STOPPING state

Communication with clients is handled via Zenoh pub/sub.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
import concurrent.futures
import struct
import signal
import numpy as np
from scipy.spatial.transform import Rotation as R, RigidTransform as Tr
import rclpy
import zenoh
import cflib
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from cflib.crazyflie import Crazyflie
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.zenoh_utils import (
    ClientStateMessage,
    HostReadyMessage,
    HostPingMessage,
    ClientPongMessage,
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


class CrazyflieWorker:
    """Worker class for managing a single Crazyflie drone in a separate process."""
    
    def __init__(self, rank: int, drone_id: int, drone_channel: int, drone_model: str,
                 ready_event: "mp.synchronize.Event", stop_event: "mp.synchronize.Event",
                 init_pose: Tr, control_mode: str, control_freq: float = 50.0,
                 start_event: "mp.synchronize.Event | None" = None,
                 failure_event: "mp.synchronize.Event | None" = None):
        """Initialize the Crazyflie worker.
        
        Args:
            rank: Drone rank/index
            drone_id: Crazyflie ID
            drone_channel: Radio channel
            drone_model: Drone model name for loading parameters
            ready_event: Event to signal when drone is connected
            stop_event: Event to signal process should stop
            init_pose: Initial pose of the drone as a RigidTransform
            control_mode: Control mode, either "attitude" or "state"
            control_freq: Control frequency in Hz
            start_event: Event to signal when all clients are ready (actions disabled until set)
            failure_event: Event to signal if connection fails (shared with other workers)
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
        
        # Configure logging for subprocess
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Drone {rank}] %(levelname)s: %(message)s',
        )
        self.logger = logging.getLogger(__name__)
        
        self.drone = None
        self.zenoh_session = None
        self.state_sub = None
        self.params = None
        self.last_action = None
        self.action_lock = threading.Lock()
    
    @staticmethod
    def _apply_drone_settings(drone: Crazyflie):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimators: 1: complementary, 2: kalman. We recommend kalman based on real-world tests
        drone.param.set_value("stabilizer.estimator", 2)
        time.sleep(0.1)  # TODO: Maybe remove
        # enable/disable tumble control. Required 0 for agressive maneuvers
        drone.param.set_value("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID; 2:Mellinger
        drone.param.set_value("stabilizer.controller", 2)
        # rate: 0, angle: 1
        drone.param.set_value("flightmode.stabModeRoll", 1)
        drone.param.set_value("flightmode.stabModePitch", 1)
        drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def _crazyflie_reset(self):
        """Reset the Crazyflie drone to a safe state."""
        # Send zero command to motors
        self.drone.platform.send_arming_request(True)
        self._apply_drone_settings(self.drone)
        pos = self.init_pose.translation
        # Reset Kalman filter values
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        yaw = self.init_pose.rotation.as_euler("xyz", degrees=False)[2]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")

    def _send_action(self, action: NDArray[np.float32]):
        """Send action command to the drone.
        
        Args:
            action: Action array, shape depends on control_mode
        """
        if self.control_mode == "attitude":
            if action.shape[0] != 4:
                raise ValueError("For attitude control, action must be of shape (4,) representing [roll, pitch, yaw, thrust]")
            pwm = force2pwm(
                action[3], self.params["thrust_max"] * 4, self.params["pwm_max"]
            )
            pwm = np.clip(pwm, self.params["pwm_min"], self.params["pwm_max"])
            action_cmd = (*np.rad2deg(action[:3]), int(pwm))
            self.drone.commander.send_setpoint(*action_cmd)
        else:  # state control
            if action.shape[0] != 13:
                raise ValueError("For state control, action must be of shape (13,)")
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )

    def _on_client_state(self, payload: str):
        """Handle client state messages containing actions."""
        msg = deserialize_message(payload, ClientStateMessage)
        
        with self.action_lock:
            self.last_action = msg.action
        
        self.logger.debug(
            f"Received action from client (gate={msg.next_gate_idx}, "
        )

    def _connect_drone(self):
        """Connect to the Crazyflie drone via radio."""
        self.logger.info(f"Connecting to drone {self.drone_id} on channel {self.drone_channel}...")
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        cflib.crtp.init_drivers()
        uri = f"radio://{self.rank}/{self.drone_channel}/2M/E7E7E7E7{self.drone_id:02X}"
        # logger.info(f"URI : {uri}")
        failure_msg: dict[str, str] = {"message": ""}
        try:
            power_switch = PowerSwitch(uri)
            power_switch.stm_power_cycle()
            wait_deadline = time.perf_counter() + 5.0
            while time.perf_counter() < wait_deadline:
                if self.stop_event.is_set():
                    raise InterruptedError("Stop requested during power-cycle wait")
                time.sleep(0.05)
        except InterruptedError as e:
            raise e
        except Exception as e:
            self.logger.error(f'{e}')
            self.failure_event.set()
            raise e

        connection_event = mp.Event()
        connection_failed_event = mp.Event()
        
        def on_connected(uri_connected):
            self.logger.info(f"Connected to {uri_connected}")
            connection_event.set()
        
        def on_connection_failed(uri_failed, msg):
            failure_msg["message"] = f"Connection failed to {uri_failed}: {msg}"
            connection_failed_event.set()
        
        def on_disconnected(uri_disconnected):
            self.logger.info(f"Disconnected from {uri_disconnected}")
        
        self.drone.fully_connected.add_callback(on_connected)
        self.drone.connection_failed.add_callback(on_connection_failed)
        self.drone.disconnected.add_callback(on_disconnected)
        
        self.drone.open_link(uri)

        timeout = 5.0
        poll_interval = 0.05
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if self.stop_event.is_set():
                raise InterruptedError("Stop requested while connecting to drone")
            if connection_failed_event.is_set():
                self.failure_event.set()
                raise RuntimeError(failure_msg["message"])
            if connection_event.is_set():
                break
            time.sleep(poll_interval)

        if not connection_event.is_set():
            raise TimeoutError(f"Timed out while waiting for the drone {self.drone_id} on channel {self.drone_channel}.")
        
        self.logger.info("Drone connected successfully")
    
    def _init_zenoh(self):
        """Initialize Zenoh session and subscribe to client state."""
        self.logger.info("Initializing Zenoh session...")
        self.zenoh_session = create_zenoh_session()
        
        # Subscribe to client state for this drone
        self.state_sub = ZenohSubscriber(
            self.zenoh_session,
            f"lsy_drone_racing/client/{self.rank}/state",
            self._on_client_state,
        )
        
        self.logger.info(f"Zenoh session for drone {self.rank} initialized")
    
    def _control_loop(self):
        """Main control loop - send actions to drone at control frequency."""
        # Wait for all clients to be ready before executing actions
        
        self.logger.info("Waiting for all clients to be ready...")
        while not self.start_event.is_set():
            if self.stop_event.is_set():
                self.logger.info("Stop event set while waiting for start event")
                return
            time.sleep(0.05)
        self.logger.info("All clients ready, starting action execution")
        
        dt = 1.0 / self.control_freq
        
        while not self.stop_event.is_set():
            start_time = time.perf_counter()
            
            with self.action_lock:
                action = self.last_action
            
            if action is not None:
                action_array = np.array(action) if isinstance(action, (list, tuple)) else np.array([action])
                self._send_action(action_array)
            
            # Maintain control frequency
            elapsed = time.perf_counter() - start_time
            if elapsed > dt:
                self.logger.warning(f"Control loop overrun: {elapsed:.3f}s, expected to be under {dt:.3f}s")
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    def _cleanup(self):
        """Clean up resources."""
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
        self.logger.info("Drone process finished")
    
    def run(self):
        """Main entry point for the worker process."""
        try:
            # Check if stop requested before doing any work
            if self.stop_event.is_set():
                self.logger.info("Stop event set before initialization, exiting immediately")
                return
            
            assert self.control_mode in ["attitude", "state"], "control_mode must be either 'attitude' or 'state'"
            
            # Load drone parameters
            self.params = load_params(physics = "first_principles", drone_model = self.drone_model)
            self.logger.info(f"Loaded parameters for {self.drone_model}")
            
            # Check if stop requested before connecting
            if self.stop_event.is_set():
                self.logger.info("Stop event set before connection attempt, exiting")
                return
            
            # Connect to drone (catch connection failures and signal other workers)
            try:
                self._connect_drone()
            except InterruptedError as e:
                # InterruptedError with stop_event set is an intentional shutdown path.
                self.logger.info(f"Connection interrupted by stop request: {e}")
                return
            except (TimeoutError, RuntimeError) as e:
                self.logger.error(f"Failed to connect to drone: {e}")
                # Signal failure to host and other workers
                if self.failure_event:
                    self.failure_event.set()
                    self.logger.info("Failure event set, notifying host to stop other workers")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during drone connection: {e}", exc_info=True)
                if self.failure_event:
                    self.failure_event.set()
                    self.logger.info("Failure event set due to unexpected error, notifying host to stop other workers")
                raise
            # Check if stop requested after connection but before reset
            if self.stop_event.is_set():
                self.logger.info("Stop event set after connection, exiting")
                return
            
            # Reset drone to initial state
            self._crazyflie_reset()
            
            # Check if stop requested before zenoh init
            if self.stop_event.is_set():
                self.logger.info("Stop event set before Zenoh initialization, exiting")
                return
            
            # Initialize Zenoh
            self._init_zenoh()
            
            # Signal ready
            self.ready_event.set()
            self.logger.info("Drone process ready")
            
            # Check if stop requested before control loop
            if self.stop_event.is_set():
                self.logger.info("Stop event set before control loop, exiting")
                return
            
            # Run control loop
            self._control_loop()
            
            self.logger.info("Drone process stopping")
        finally:
            self._cleanup()
            


def crazyflie_process_worker(
    rank: int,
    drone_id: int,
    drone_channel: int,
    drone_model: str,
    ready_event: "mp.synchronize.Event",
    stop_event: "mp.synchronize.Event",
    init_pose: Tr,
    control_mode: str,
    start_event: "mp.synchronize.Event",
    failure_event: "mp.synchronize.Event | None" = None,
    control_freq: float = 50.0,
    
):
    """Entry point for Crazyflie worker process.
    
    This function is called by multiprocessing and creates a CrazyflieWorker instance.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker = CrazyflieWorker(
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
    )
    worker.run()


class RealRaceHostState(Enum):
    """States of the RealRaceHost.
    
    Attributes:
        IDLE: Loading track configuration and checking track/drone positions.
        INITIALIZED: Connected to drones, waiting for clients to be ready.
        OPERATION: Race in progress, supervising clients.
        STOPPING: Shutting down gracefully.
    """
    IDLE = 0
    INITIALIZED = 1
    OPERATION = 2
    STOPPING = 3


@dataclass
class DroneConnection:
    """Information about a drone connection managed by the host."""
    rank: int
    drone_id: int
    drone_channel: int
    process: mp.Process | None = None
    ready_event: "mp.synchronize.Event | None" = None
    stop_event: "mp.synchronize.Event | None" = None
    failure_event: "mp.synchronize.Event | None" = None
    connected: bool = False


class RealRaceHost:
    """Base host class for real-world drone racing.
    
    This class coordinates multi-drone races by managing track validation, drone connections,
    and client synchronization via Zenoh communication.
    """
    
    state: RealRaceHostState
    _num_drones: int = 0
    _config: ConfigDict | None = None
    
    _zenoh_session: zenoh.Session | None
    _host_ready_pub: ZenohPublisher | None
    _race_start_pub: ZenohPublisher | None
    
    _client_state_subs: dict[int, ZenohSubscriber] | None

    def __init__(self, config: ConfigDict):
        """Initialize the RealRaceHost.
        
        Args:
            config: Configuration dictionary containing deploy, env, and sim settings.
        """
        self.state = RealRaceHostState.IDLE
        self._config = config
        self._shutdown_event = threading.Event()
        self._clients_ready: dict[int, bool] = {}
        self._clients_stopped: dict[int, bool] = {}
        self._start_time = time.perf_counter()

        self._host_ready_pub = None
        self._race_start_pub = None
        self._client_state_subs = None
        
        self.load_config(config)
        self.init_zenoh()

    def init_zenoh(self, conf: zenoh.Config | None = None) -> zenoh.Session:
        """Initialize Zenoh communication."""
        self._client_state_subs = {}
        self._zenoh_session = create_zenoh_session(conf)
        
        # Create publishers
        self._host_ready_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/ready")
        self._host_ping_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/ping")
        self._race_start_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/race_start")
        
        # Subscribe to client state messages for readiness detection and stopped detection
        for rank in range(self._num_drones):
            def on_client_state(payload: str, rank=rank):
                msg = deserialize_message(payload, ClientStateMessage)
                
                # Mark client as ready on first state message
                if not self._clients_ready[rank]:
                    logger.debug(f"Client {rank} ready (received first state message")
                    self._clients_ready[rank] = True
                
                # Track stopped clients
                if msg.stopped:
                    logger.info(f"Client {rank} stopped (next_gate_idx={msg.next_gate_idx}")
                    self._clients_stopped[rank] = True
            
            sub = ZenohSubscriber(
                self._zenoh_session,
                f"lsy_drone_racing/client/{rank}/state",
                on_client_state,
            )
            self._client_state_subs[rank] = sub
        
        # Subscribe to pong messages for clock offset calibration
        self._client_pong_subs = {}
        for rank in range(self._num_drones):
            def on_client_pong(payload: str, rank=rank):
                msg = deserialize_message(payload, ClientPongMessage)
                # Calculate round-trip time
                rtt = time.perf_counter() - msg.host_timestamp
                # Clock offset = (RTT / 2) + (client_timestamp - host_timestamp)
                # This accounts for the network delay and clock difference
                half_rtt = rtt / 2.0
                clock_offset = (msg.client_timestamp - msg.host_timestamp) - half_rtt
                self._client_clock_offsets[rank] = clock_offset
                logger.info(f"Client {rank} clock offset calibrated: {clock_offset*1000:.2f}ms (RTT: {rtt*1000:.2f}ms)")
            
            sub = ZenohSubscriber(
                self._zenoh_session,
                f"lsy_drone_racing/client/{rank}/pong",
                on_client_pong,
            )
            self._client_pong_subs[rank] = sub
        
        logger.info("Zenoh communication initialized")
        return self._zenoh_session
    
    def load_config(self, config: ConfigDict):
        """Load configuration.
        
        Args:
            config: Configuration dictionary.
        """
        raise NotImplementedError("Subclass must implement load_config")
    
    def connect_drones(self):
        """Connect to all drones."""
        raise NotImplementedError("Subclass must implement connect_drones")
    
    def host_main_loop(self):
        """Main loop of the host."""
        raise NotImplementedError("Subclass must implement host_main_loop")

    def close(self):
        """Gracefully close all resources and shutdown the host.
        
        This method can be called at any point during the host lifecycle
        and will properly clean up resources.
        """
        raise NotImplementedError("Subclass must implement close")


class CrazyFlieRealRaceHost(RealRaceHost):
    """LSY's implementation of RealRaceHost for multi-drone racing with Crazyflies."""
    
    _drone_names: list[str]
    _drone_ids: list[int]
    _drone_channels: list[int]
    _drone_models: list[str]
    _drone_connections: dict[int, DroneConnection] | None
    _drone_control_freq: list[float]
    _all_clients_ready_event: "mp.synchronize.Event | None"
    _mp_ctx: mp.context.BaseContext
    _client_clock_offsets : dict[int, float]
    

    def __init__(self, config: ConfigDict):
        """Initialize LSYRealRaceHost."""
        super().__init__(config)
        self._mp_ctx = mp.get_context("spawn")
        self._all_clients_ready_event = self._mp_ctx.Event()
        self._drone_connections = None  # Initialize early so close() can access it
        self._client_clock_offsets = {}  # Store clock offsets for each client
        
        logger.info("Host: In IDLE state - checking track...")
        # TODO: Testing the pipeline without checking the track
        # self.check_track(rng_config = config.env.randomizations)
    
    def load_config(self, config: ConfigDict):
        """Load track configuration."""
        self._config = config
        self.gates, self.obstacles, self.drones_track = load_track(config.env.track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(config.env.track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(config.env.track.safety_limits["pos_limit_high"])
        
        self._num_drones = len(config.deploy.drones)
        self._drone_names = [f"cf{drone['id']}" for drone in config.deploy.drones]
        self._drone_ids = [drone['id'] for drone in config.deploy.drones]
        self._drone_channels = [drone['channel'] for drone in config.deploy.drones]
        self._drone_models = [drone['drone_model'] for drone in config.deploy.drones]
        self._drone_control_freq = [kwargs['freq'] for kwargs in config.env.kwargs]
        
        # Initialize client tracking
        for rank in range(self._num_drones):
            self._clients_ready[rank] = False
            self._clients_stopped[rank] = False
    
    def check_track(self, rng_config: ConfigDict):
        """Check and validate the track."""
        logger.info("Host: Checking track configuration...")
        
        # Initialize ROS connectors in parallel
        def init_track_connector():
            """Initialize connector for track objects."""
            tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
            tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
            try:
                ros_connector = ROSConnector(tf_names=tf_names, timeout=10.0)
                return ros_connector
            
            except Exception as e:
                return e
        
        def init_drone_connector():
            """Initialize connector for drone positions."""
            try:
                ros_connector = ROSConnector(tf_names=self._drone_names, timeout=10.0)
                return ros_connector
            except Exception as e:
                return e
        
        logger.info("Host: Initializing ROS connectors in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_track = executor.submit(init_track_connector)
            future_drones = executor.submit(init_drone_connector)
            
            ros_connector_track = future_track.result()
            ros_connector_drones = future_drones.result()
        
        if isinstance(ros_connector_track, Exception):
            raise ros_connector_track
        if isinstance(ros_connector_drones, Exception):
            raise ros_connector_drones
        
        # Update track poses from motion capture
        self._update_track_poses(ros_connector_track)
        
        # Check race track
        check_race_track(
            gates_pos=self.gates.pos,
            nominal_gates_pos=self.gates.nominal_pos,
            gates_quat=self.gates.quat,
            nominal_gates_quat=self.gates.nominal_quat,
            obstacles_pos=self.obstacles.pos,
            nominal_obstacles_pos=self.obstacles.nominal_pos,
            rng_config=rng_config,
        )
        
        # Check drone start positions
        for rank, drone_name in enumerate(self._drone_names):
            check_drone_start_pos(
                nominal_pos=self.drones_track.pos[rank],
                real_pos=ros_connector_drones.pos[drone_name],
                rng_config=rng_config,
                drone_name=drone_name,
            )
        
        # Close temporary connectors
        ros_connector_track.close()
        ros_connector_drones.close()
        
        logger.info("Host: Track check passed")
    
    def connect_drones(self):
        """Connect to all Crazyflie drones by spawning individual processes.
        
        Uses non-blocking polling to wait for connections, allowing responsive
        shutdown via stop_event during the connection phase.
        """
        logger.info(f"Host: Spawning processes for {self._num_drones} Crazyflie drones...")
        self._drone_connections = {}
        
        # Create shared failure event for all workers
        failure_event = self._mp_ctx.Event()
        
        for rank in range(self._num_drones):
            drone_id = self._drone_ids[rank]
            channel = self._drone_channels[rank]
            drone_model = self._drone_models[rank]
            control_freq = self._drone_control_freq[rank]
            init_pose = Tr.from_components(translation=self.drones_track.pos[rank],rotation = R.from_quat(self.drones_track.quat[rank]))
            control_mode = self._config.env.kwargs[rank]['control_mode']
            
            # Create synchronization events
            ready_event = self._mp_ctx.Event()
            stop_event = self._mp_ctx.Event()
            
            # Spawn process
            process = self._mp_ctx.Process(
                target=crazyflie_process_worker,
                args=(rank,
                    drone_id,
                    channel,
                    drone_model,
                    ready_event,
                    stop_event,
                    init_pose,
                     control_mode,
                     self._all_clients_ready_event,
                     failure_event,
                     control_freq),
                name=f"CrazyflieProcess-{rank}",
            )
            process.start()
            
            # Store connection info
            conn = DroneConnection(
                rank=rank,
                drone_id=drone_id,
                drone_channel=channel,
                process=process,
                ready_event=ready_event,
                stop_event=stop_event,
                failure_event=failure_event,
                connected=False,
            )
            self._drone_connections[rank] = conn
            
            logger.info(f"Spawned process for Crazyflie {rank} (PID: {process.pid})")
        
        # Non-blocking polling for process readiness
        logger.info("Waiting for all Crazyflie processes to be ready...")
        timeout = 30  # seconds
        poll_interval = 0.5  # check every 500ms
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout:
            # Check if any worker has failed
            if failure_event.is_set():
                logger.error("Worker failure detected! Signaling all workers to stop...")
                # Signal all workers to stop
                for conn in self._drone_connections.values():
                    conn.stop_event.set()
                raise RuntimeError("Connection failed in one of the Crazyflie workers. Stopping all workers.")
            
            # Check if all processes are ready
            all_ready = all(conn.ready_event.is_set() for conn in self._drone_connections.values())
            if all_ready:
                for conn in self._drone_connections.values():
                    conn.connected = True
                    logger.info(f"Crazyflie {conn.rank} is ready")
                logger.info(f"Host: All {self._num_drones} Crazyflie processes are ready")
                self.state = RealRaceHostState.INITIALIZED
                return
            
            # Check if any process has died unexpectedly
            for rank, conn in self._drone_connections.items():
                if conn.process and not conn.process.is_alive() and not conn.ready_event.is_set():
                    logger.error(f"Crazyflie process {rank} died unexpectedly without signaling ready")
                    # Signal all workers to stop
                    for other_conn in self._drone_connections.values():
                        other_conn.stop_event.set()
                    raise RuntimeError(f"Process {rank} terminated unexpectedly during connection phase")
            
            # Poll with small delay to allow responsiveness to signals
            time.sleep(poll_interval)
        
        # Timeout occurred
        logger.error(f"Timeout waiting for Crazyflie processes to become ready")
        # Signal all workers to stop
        for conn in self._drone_connections.values():
            conn.stop_event.set()
        raise TimeoutError(f"Timeout waiting for all {self._num_drones} Crazyflie processes to connect")
    
    def _update_track_poses(self, ros_connector: ROSConnector):
        """Update track poses from motion capture system.
        
        Args:
            ros_connector: ROSConnector with track object tf_names.
        """
        tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
        for i, tf_name in enumerate(tf_names):
            try:
                pos = ros_connector.pos[tf_name]
                quat = ros_connector.quat[tf_name]
                self.gates.pos[i] = pos
                self.gates.quat[i] = quat
            except Exception:
                logger.warning(f"Could not update pose for {tf_name}")
        
        tf_names = [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
        for i, tf_name in enumerate(tf_names):
            try:
                pos = ros_connector.pos[tf_name]
                quat = ros_connector.quat[tf_name]
                self.obstacles.pos[i] = pos
                self.obstacles.quat[i] = quat
            except Exception:
                logger.warning(f"Could not update pose for {tf_name}")
    
    def close(self):
        """Gracefully close all Crazyflie-specific resources and shutdown.
        
        This method performs shutdown in phases:
        1. Signal all drone processes to stop
        2. Wait for drone processes to finish with timeout
        3. Terminate any unresponsive processes
        4. Close Zenoh communication
        """
        logger.info("Host shutting down...")
        
        # Phase 1: Signal all drone processes to stop
        if self._drone_connections is not None:
            logger.info("Phase 1: Signaling all Crazyflie processes to stop...")
            for conn in self._drone_connections.values():
                if conn.stop_event:
                    conn.stop_event.set()
            
            # Phase 2: Wait for drone processes to finish with reasonable timeout
            logger.info("Phase 2: Waiting for Crazyflie processes to finish...")
            max_wait_per_process = 5  # seconds
            for rank, conn in self._drone_connections.items():
                if conn.process and conn.process.is_alive():
                    conn.process.join(timeout=max_wait_per_process)
                    if conn.process.is_alive():
                        logger.warning(f"Process {rank} did not terminate, terminating...")
                        conn.process.terminate()
                        conn.process.join(timeout=2)
                        if conn.process.is_alive():
                            logger.warning(f"Process {rank} still alive after terminate, killing...")
                            conn.process.kill()
                            conn.process.join()
                    else:
                        logger.info(f"Process {rank} terminated successfully")
        
        # Phase 3: Close Zenoh communication
        logger.info("Phase 3: Closing Zenoh communication...")
        if self._host_ready_pub:
            self._host_ready_pub.close()
        if self._host_ping_pub:
            self._host_ping_pub.close()
        if self._race_start_pub:
            self._race_start_pub.close()
        if self._client_state_subs:
            for sub in self._client_state_subs.values():
                sub.close()
        if self._client_pong_subs:
            for sub in self._client_pong_subs.values():
                sub.close()
        if self._zenoh_session:
            self._zenoh_session.close()
        
        logger.info("Host shutdown complete")
    
    
    
    
    def _calibrate_client_clocks(self):
        """Calibrate clock offsets with all clients via ping-pong.
        
        This should be called after all clients are ready but before the race starts.
        """
        logger.info("Host: Calibrating client clocks via ping-pong...")
        self._client_clock_offsets = {}
        
        # Send pings to all clients
        for rank in range(self._num_drones):
            msg = HostPingMessage(drone_rank=rank, host_timestamp=time.perf_counter())
            self._host_ping_pub.publish(msg)
        
        # Wait for all pongs with timeout
        timeout = 10.0  # seconds
        poll_interval = 0.01  # 10ms
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout:
            if len(self._client_clock_offsets) == self._num_drones:
                logger.info(f"Host: All {self._num_drones} clients calibrated")
                return
            time.sleep(poll_interval)
        
        logger.warning(f"Host: Only {len(self._client_clock_offsets)} / {self._num_drones} clients calibrated within timeout")
    
    def host_main_loop(self, race_update_freq : float = 50.0):
        """Main loop of the host."""
        logger.info("Host: Starting main loop")
        
        if self.state != RealRaceHostState.INITIALIZED:
            raise RuntimeError("Host: Failed to reach INITIALIZED state")
        
        # INITIALIZED state: Wait for clients to be ready
        logger.info("Host: In INITIALIZED state - waiting for clients...")
        host_ready_freq = 10  # Hz
        timeout = 30  # seconds
        t_start = time.perf_counter()
        
        while time.perf_counter() - t_start < timeout:
            # Send host ready messages
            msg = HostReadyMessage(elapsed_time=0.0, timestamp=time.perf_counter())
            self._host_ready_pub.publish(msg)
            
            # Check if all clients are ready
            if all(self._clients_ready.values()):
                logger.info("Host: All clients are ready!")
                break
            
            time.sleep(1.0 / host_ready_freq)
        
        if not all(self._clients_ready.values()):
            raise TimeoutError("Host: Timeout waiting for all clients to become ready")
        
        # Calibrate client clocks
        self._calibrate_client_clocks()
        
        # Signal drone processes that all clients are ready
        logger.info("Host: Signaling drone processes that all clients are ready")
        self._all_clients_ready_event.set()
        
        # OPERATION state: Race started
        logger.info("Host: In OPERATION state - race started")
        self.state = RealRaceHostState.OPERATION
        self._start_time = time.perf_counter()
        
        while self.state == RealRaceHostState.OPERATION:
            elapsed_time = time.perf_counter() - self._start_time
            
            # Send periodic race start messages
            finished = all(self._clients_stopped.values())
            msg = RaceStartMessage(
                elapsed_time=elapsed_time,
                timestamp=time.perf_counter(),
                finished=finished,
            )
            self._race_start_pub.publish(msg)
            
            if finished:
                logger.info("Host: All clients stopped, transitioning to STOPPING state")
                self.state = RealRaceHostState.STOPPING
                break
            
            time.sleep(1.0 / race_update_freq)
        
        
    
        

