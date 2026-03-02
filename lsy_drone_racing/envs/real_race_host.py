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
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
import concurrent.futures

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
                 start_event: "mp.synchronize.Event | None" = None):
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
            all_clients_ready_event: Event to signal when all clients are ready (actions disabled until set)
        """
        self.rank = rank
        self.drone_id = drone_id
        self.drone_channel = drone_channel
        self.drone_model = drone_model
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.start_event = start_event
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
        latency_ms = compute_latency_ms(msg.timestamp)
        
        with self.action_lock:
            self.last_action = msg.action
        
        self.logger.debug(
            f"Received action from client (gate={msg.next_gate_idx}, "
            f"latency={latency_ms:.2f}ms)"
        )

    def _connect_drone(self):
        """Connect to the Crazyflie drone via radio."""
        self.logger.info(f"Connecting to drone {self.drone_id} on channel {self.drone_channel}...")
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        cflib.crtp.init_drivers()
        uri = f"radio://0/{self.drone_channel}/2M/E7E7E7E7E{self.drone_id:02X}"

        power_switch = PowerSwitch(uri)
        power_switch.stm_power_cycle()
        time.sleep(2)

        connection_event = mp.Event()
        
        def on_connected(uri_connected):
            self.logger.info(f"Connected to {uri_connected}")
            connection_event.set()
        
        def on_connection_failed(uri_failed, msg):
            raise RuntimeError(f"Connection failed to {uri_failed}: {msg}")
        
        def on_disconnected(uri_disconnected):
            self.logger.info(f"Disconnected from {uri_disconnected}")
        
        self.drone.fully_connected.add_callback(on_connected)
        self.drone.connection_failed.add_callback(on_connection_failed)
        self.drone.disconnected.add_callback(on_disconnected)
        
        self.drone.open_link(uri)
        
        if not connection_event.wait(timeout=30):
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
        
        self.logger.info("Zenoh session initialized")
    
    def _control_loop(self):
        """Main control loop - send actions to drone at control frequency."""
        # Wait for all clients to be ready before executing actions
        
        self.logger.info("Waiting for all clients to be ready...")
        self.start_event.wait()
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
            # Send zero command before disconnecting
            self.drone.commander.send_setpoint(0, 0, 0, 0)
            self.drone.close_link()
        
        self.logger.info("Drone process finished")
    
    def run(self):
        """Main entry point for the worker process."""
        try:
            assert self.control_mode in ["attitude", "state"], "control_mode must be either 'attitude' or 'state'"
            
            # Load drone parameters
            self.params = load_params(self.drone_model)
            self.logger.info(f"Loaded parameters for {self.drone_model}")
            
            # Connect to drone
            self._connect_drone()
            
            # Reset drone to initial state
            self._crazyflie_reset()
            
            # Initialize Zenoh
            self._init_zenoh()
            
            # Signal ready
            self.ready_event.set()
            self.logger.info("Drone process ready")
            
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
    control_freq: float = 50.0,
    
):
    """Entry point for Crazyflie worker process.
    
    This function is called by multiprocessing and creates a CrazyflieWorker instance.
    """
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
    connected: bool = False


class RealRaceHost:
    """Base host class for real-world drone racing.
    
    This class coordinates multi-drone races by managing track validation, drone connections,
    and client synchronization via Zenoh communication.
    """
    
    state: RealRaceHostState
    _num_drones: int = 0
    _config: ConfigDict | None = None
    
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
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.load_config(config)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, transitioning to STOPPING state")
        self.state = RealRaceHostState.STOPPING
        self._shutdown_event.set()
    
    def load_config(self, config: ConfigDict):
        """Load configuration.
        
        Args:
            config: Configuration dictionary.
        """
        raise NotImplementedError("Subclass must implement load_config")
    
    def connect_drones(self):
        """Connect to all drones."""
        raise NotImplementedError("Subclass must implement connect_drones")
    
    def init_zenoh(self, conf: zenoh.Config | None = None) -> zenoh.Session:
        """Initialize Zenoh session and communication.
        
        Args:
            conf: Optional Zenoh configuration.
            
        Returns:
            Zenoh session.
        """
        raise NotImplementedError("Subclass must implement init_zenoh")
    
    def host_main_loop(self):
        """Main loop of the host."""
        raise NotImplementedError("Subclass must implement host_main_loop")


class CrazyFlieRealRaceHost(RealRaceHost):
    """LSY's implementation of RealRaceHost for multi-drone racing with Crazyflies."""
    
    _drone_names: list[str]
    _drone_ids: list[int]
    _drone_channels: list[int]
    _drone_models: list[str]
    _drone_connections: dict[int, DroneConnection]
    _drone_control_freq: list[float]
    _all_clients_ready_event: "mp.synchronize.Event | None"

    _zenoh_session: zenoh.Session | None
    _host_ready_pub: ZenohPublisher | None
    _race_start_pub: ZenohPublisher | None
    
    _client_ready_subs: dict[int, ZenohSubscriber]
    _client_state_subs: dict[int, ZenohSubscriber]

    def __init__(self, config: ConfigDict):
        """Initialize LSYRealRaceHost."""
        super().__init__(config)
        self.gates = None
        self.obstacles = None
        self.drones_track = None
        self.n_gates = 0
        self.n_obstacles = 0
        self.pos_limit_low = None
        self.pos_limit_high = None
        self._all_clients_ready_event = mp.Event()
        
        logger.info("Host: In IDLE state - checking track...")
        # TODO: Testing the pipeline without checking the track
        # self.check_track(rng_config = config.env.randomizations)   
        logger.info("Host: Connecting to drones...")
        self.connect_drones()
    
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
        self._drone_control_freq = [drone['freq'] for drone in config.env.kwargs.freq]
        
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
        """Connect to all Crazyflie drones by spawning individual processes."""
        logger.info(f"Host: Spawning processes for {self._num_drones} Crazyflie drones...")
        self._drone_connections = {}
        
        for rank in range(self._num_drones):
            drone_id = self._drone_ids[rank]
            channel = self._drone_channels[rank]
            drone_model = self._drone_models[rank]
            control_freq = self._drone_control_freq[rank]
            init_pose = Tr(self.drones_track.pos[rank], R.from_quat(self.drones_track.quat[rank]))
            control_mode = self._config.env.kwargs.control_mode
            
            # Create synchronization events
            ready_event = mp.Event()
            stop_event = mp.Event()
            
            # Spawn process
            process = mp.Process(
                target=crazyflie_process_worker,
                args=(rank, drone_id, channel, drone_model, ready_event, stop_event, init_pose, control_mode, control_freq, self._all_clients_ready_event),
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
                connected=False,
            )
            self._drone_connections[rank] = conn
            
            logger.info(f"Spawned process for Crazyflie {rank} (PID: {process.pid})")
        
        # Wait for all processes to be ready
        logger.info("Waiting for all Crazyflie processes to be ready...")
        timeout = 30  # seconds
        
        for rank, conn in self._drone_connections.items():
            if not conn.ready_event.wait(timeout=timeout):
                raise TimeoutError(f"Timeout waiting for Crazyflie {rank} to be ready")
            else:
                conn.connected = True
                logger.info(f"Crazyflie {rank} is ready")
        
        if len(self._drone_connections) == self._num_drones:
            logger.info(f"Host: All {self._num_drones} Crazyflie processes are ready")
            self.state = RealRaceHostState.INITIALIZED
        else:
            raise RuntimeError(
                f"Host: Only {sum(c.connected for c in self._drone_connections.values())}/{self._num_drones} Crazyflie processes ready"
            )
    
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
    
    def _cleanup(self):
        """Clean up Crazyflie-specific resources."""
        # Signal drone processes to stop
        logger.info("Signaling Crazyflie processes to stop...")
        for conn in self._drone_connections.values():
            if conn.stop_event:
                conn.stop_event.set()
        
        # Wait for processes to finish
        logger.info("Waiting for Crazyflie processes to finish...")
        for rank, conn in self._drone_connections.items():
            if conn.process and conn.process.is_alive():
                conn.process.join(timeout=5)
                if conn.process.is_alive():
                    logger.warning(f"Crazyflie process {rank} did not terminate, killing...")
                    conn.process.terminate()
                    conn.process.join(timeout=2)
        
        # Close Zenoh publishers/subscribers
        if self._host_ready_pub:
            self._host_ready_pub.close()
        if self._race_start_pub:
            self._race_start_pub.close()
        for sub in self._client_state_subs.values():
            sub.close()
        
        # Close Zenoh session
        if self._zenoh_session:
            self._zenoh_session.close()
        
        # Call parent cleanup
        super()._cleanup()
        
        logger.info("Host: CrazyflieRaceHost cleanup complete")
    
    def init_zenoh(self, conf: zenoh.Config | None = None) -> zenoh.Session:
        """Initialize Zenoh communication."""
        self._client_state_subs = {}
        self._zenoh_session = create_zenoh_session(conf)
        
        # Create publishers
        self._host_ready_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/ready")
        self._race_start_pub = ZenohPublisher(self._zenoh_session, "lsy_drone_racing/host/race_start")
        
        # Subscribe to client state messages for readiness detection and stopped detection
        for rank in range(self._num_drones):
            def on_client_state(payload: str, rank=rank):
                msg = deserialize_message(payload, ClientStateMessage)
                latency_ms = compute_latency_ms(msg.timestamp)
                
                # Mark client as ready on first state message
                if not self._clients_ready[rank]:
                    logger.debug(f"Client {rank} ready (received first state message, latency: {latency_ms:.2f}ms)")
                    self._clients_ready[rank] = True
                
                # Track stopped clients
                if msg.stopped:
                    logger.info(f"Client {rank} stopped (next_gate_idx={msg.next_gate_idx}, latency: {latency_ms:.2f}ms)")
                    self._clients_stopped[rank] = True
            
            sub = ZenohSubscriber(
                self._zenoh_session,
                f"lsy_drone_racing/client/{rank}/state",
                on_client_state,
            )
            self._client_state_subs[rank] = sub
        
        logger.info("Zenoh communication initialized")
        return self._zenoh_session
    
    def host_main_loop(self, race_update_freq : float = 50.0):
        """Main loop of the host."""
        logger.info("Host: Starting main loop")
        
        if self.state != RealRaceHostState.INITIALIZED:
            raise RuntimeError("Host: Failed to reach INITIALIZED state")
        
        # Initialize Zenoh
        self.init_zenoh()
        
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
        
        
    
        

