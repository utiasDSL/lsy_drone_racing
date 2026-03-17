"""Client-side environment for multi-drone racing with host-client architecture.

The RealMultiDroneRaceEnvClient operates as a client in a host-client system:
- Receives coordination messages from the host via Zenoh
- Manages a single drone's state and control
- Sends state updates to the host for supervision
- Handles local observation and gate tracking
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np
import rclpy
import zenoh
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from gymnasium import Env
from scipy.spatial.transform import Rotation as R
import concurrent.futures

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.zenoh_utils import (
    ClientStateMessage,
    HostReadyMessage,
    HostInitializedMessage,
    ClientPingMessage,
    HostPongMessage,
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
    """Data structure for client-side environment state."""
    
    def __init__(self, n_drones: int, n_gates: int, n_obstacles: int):
        """Initialize client environment data.
        
        Args:
            n_drones: Number of drones in the race.
            n_gates: Number of gates in the track.
            n_obstacles: Number of obstacles in the track.
        """
        self.n_drones = n_drones
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        
        self.target_gate = np.zeros(n_drones, dtype=int)
        self.gates_visited = np.zeros((n_drones, n_gates), dtype=bool)
        self.obstacles_visited = np.zeros((n_drones, n_obstacles), dtype=bool)
        self.last_drone_pos = np.zeros((n_drones, 3), dtype=np.float32)
        
        self.taken_off = False
    
    def reset(self, last_drone_pos: NDArray[np.float32]):
        """Reset the environment data."""
        self.target_gate[:] = 0
        self.gates_visited[:] = False
        self.obstacles_visited[:] = False
        self.last_drone_pos[:] = last_drone_pos
        self.taken_off = False


class RealMultiDroneRaceEnvClient(Env):
    """Client-side environment for multi-drone racing.
    
    This environment is designed to run on each drone's computing unit, communicating with
    the central host via Zenoh for coordination and supervision.
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
            drones: List of all drones in the race.
            rank: Rank of this drone.
            freq: Control frequency.
            track: Track configuration.
            randomizations: Randomization configuration.
            sensor_range: Sensor range for gates/obstacles.
            control_mode: "state" or "attitude" control.
        """
        # Basic setup
        self.n_drones = len(drones)
        self.rank = rank
        self.freq = freq
        self.sensor_range = sensor_range
        self.control_mode = control_mode
        
        # Load drone info
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]

        # Load track
        self.gates, self.obstacles, self.drones_track = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])
        
        # Initialize JAX
        self.device = jax.devices("cpu")[0]
        
        # Initialize ROS connectors
        # High-precision estimator for own drone
        self._ros_connector_own = None
        # TF-based connector for other drones
        self._ros_connector_others = None
        
        # Initialize environment data
        self.data = ClientEnvData(self.n_drones, self.n_gates, self.n_obstacles)
        
        # Zenoh communication
        self._zenoh_session: zenoh.Session | None = None
        self._host_ready_sub: ZenohSubscriber | None = None
        self._host_ping_sub: ZenohSubscriber | None = None
        self._race_start_sub: ZenohSubscriber | None = None
        self._client_state_pub: ZenohPublisher | None = None
        self._client_pong_pub: ZenohPublisher | None = None
        
        # Race state
        self._host_ready = False
        self._host_ready_event = threading.Event()
        self._race_started = False
        self._race_start_time = 0.0
        self._should_stop = False
        self._last_host_elapsed_time = 0.0
        self._last_race_start_timestamp = 0.0
        self._clock_offset = 0.0  # Will be set during calibration
        
        self._jit_compiled = False
    
    def _init_ros_connectors(self):
        """Initialize ROS connectors in parallel for speed."""
        def init_own_connector():
            """Initialize high-precision estimator connector for own drone."""
            return ROSConnector(
                estimator_names=[self.drone_name],
                cmd_topic=f"/drones/{self.drone_name}/command",
                timeout=10.0,
            )
        
        def init_others_connector():
            """Initialize TF-based connector for other drones."""
            other_drone_names = [name for i, name in enumerate(self.drone_names) if i != self.rank]
            if not other_drone_names:
                return None
            return ROSConnector(
                tf_names=other_drone_names,
                timeout=10.0,
            )
        
        logger.info(f"Client {self.rank}: Initializing ROS connectors for drones in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_own = executor.submit(init_own_connector)
            future_others = executor.submit(init_others_connector)
            self._ros_connector_own = future_own.result()
            self._ros_connector_others = future_others.result()
        
        
        logger.info(f"Client {self.rank}: ROS connectors initialized")
    
    
    def _get_all_drone_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Get full state of all drones.
        
        Returns:
            Tuple of (positions, quaternions).
        """
        pos = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        quat = np.full((self.n_drones, 4), np.nan, dtype=np.float32)
        ang_vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        # Own drone from high-precision estimator
        pos[self.rank] = self._ros_connector_own.pos[self.drone_name]
        quat[self.rank] = self._ros_connector_own.quat[self.drone_name]
        vel[self.rank] = self._ros_connector_own.vel[self.drone_name]
        ang_vel[self.rank] = self._ros_connector_own.ang_vel[self.drone_name]
        
        
        # Other drones from TF
        if self._ros_connector_others is not None:
            for i, name in enumerate(self.drone_names):
                if i != self.rank:
                    pos[i] = self._ros_connector_others.pos[name]
                    quat[i] = self._ros_connector_others.quat[name]
                    # vel[i] = self._ros_connector_others.vel[name]
                    # ang_vel[i] = self._ros_connector_others.ang_vel[name]

        return pos, quat, vel, ang_vel
    
    def _init_zenoh(self):
        """Initialize Zenoh communication."""
        self._zenoh_session = create_zenoh_session()
        
        # Subscribe to host messages
        def on_host_ready(payload: str):
            try:
                msg = deserialize_message(payload, HostReadyMessage)
                self._host_ready = True
                self._host_ready_event.set()  # Signal waiting thread
                latency_ms = compute_latency_ms(msg.timestamp)
               
                logger.debug(f"Client {self.rank}: Received host ready (latency: {latency_ms:.2f}ms)")
            except Exception as e:
                logger.error(f"Error processing host ready message: {e}")
        
        self._host_ready_sub = ZenohSubscriber(
            self._zenoh_session,
            "lsy_drone_racing/host/ready",
            on_host_ready,
        )
        
        def on_host_initialized(payload: str):
            """Handle initialized message from host - trigger clock calibration."""
            try:
                msg = deserialize_message(payload, HostInitializedMessage)
                if msg.drone_rank == self.rank:
                    # Send ping immediately with client timestamp
                    ping_msg = ClientPingMessage(
                        drone_rank=self.rank,
                        client_timestamp=time.time(),
                    )
                    self._client_ping_pub.publish(ping_msg)
                    logger.debug(f"Client {self.rank}: Sent ping for clock calibration")
            except Exception as e:
                logger.error(f"Error processing host initialized message: {e}")
        
        self._host_initialized_sub = ZenohSubscriber(
            self._zenoh_session,
            "lsy_drone_racing/host/initialized",
            on_host_initialized,
        )
        
        def on_host_pong(payload: str):
            """Handle pong from host - calculate and store clock offset."""
            try:
                msg = deserialize_message(payload, HostPongMessage)
                if msg.drone_rank == self.rank:
                    # Calculate clock offset for this client
                    # offset = (host_time - client_time) - RTT/2
                    # Since we're measuring RTT now, we approximate:
                    # offset = host_time - client_time (measured at approximately same moment)
                    # The RTT is typically small (~1-10ms for local/nearby machines)
                    self._clock_offset = float(msg.host_timestamp) - time.time()
                    logger.info(f"Client {self.rank}: Clock offset calibrated: {self._clock_offset*1000:.2f}ms")
            except Exception as e:
                logger.error(f"Error processing host pong message: {e}")
        
        self._host_pong_sub = ZenohSubscriber(
            self._zenoh_session,
            "lsy_drone_racing/host/pong",
            on_host_pong,
        )
        
        def on_race_start(payload: str):
            try:
                msg = deserialize_message(payload, RaceStartMessage)
                self._race_started = True
                self._race_start_time = time.time() - msg.elapsed_time
                self._last_host_elapsed_time = msg.elapsed_time
                self._last_race_start_timestamp = msg.timestamp  # Store for echo
                latency_ms = compute_latency_ms(msg.timestamp)
            except Exception as e:
                logger.error(f"Error processing race start message: {e}")
        
        self._race_start_sub = ZenohSubscriber(
            self._zenoh_session,
            "lsy_drone_racing/host/race_start",
            on_race_start,
        )
        
        # Create publishers
        self._client_state_pub = ZenohPublisher(
            self._zenoh_session,
            f"lsy_drone_racing/client/{self.rank}/state",
        )
        
        self._client_ping_pub = ZenohPublisher(
            self._zenoh_session,
            f"lsy_drone_racing/client/{self.rank}/ping",
        )
        
        logger.info(f"Client {self.rank}: Zenoh communication initialized")
    
    def _jit(self):
        """Compile JAX functions."""
        if self._jit_compiled:
            return
        
        # Dummy call to compile gate_passed function
        try:
            dummy_drone_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
            dummy_gate_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
            dummy_gate_quat = np.zeros((self.n_drones, 4), dtype=np.float32)
            
            with jax.default_device(self.device):
                gate_passed(
                    dummy_drone_pos,
                    dummy_drone_pos,
                    dummy_gate_pos,
                    dummy_gate_quat,
                    (0.45, 0.45),
                )
            self._jit_compiled = True
        except Exception as e:
            logger.warning(f"Client {self.rank}: JAX compilation warning: {e}")
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment.
        
        Args:
            seed: Random seed (unused in real environment).
            options: Options dictionary from deployment.
            
        Returns:
            Initial observation and info.
        """
        options = {} if options is None else options
                
        # Initialize ROS connectors if not already done
        if self._ros_connector_own is None:
            self._init_ros_connectors()
        
        # Initialize Zenoh if not already done
        if self._zenoh_session is None:
            self._init_zenoh()
        
        # Get current drone positions
        current_positions, _, _, _ = self._get_all_drone_states()
        self.data.reset(current_positions)
        
        logger.info(f"Client {self.rank}: Waiting for host ready message...")
        max_wait_time = 120.0
        
        # Send state messages in background thread to signal readiness
        stop_sending = threading.Event()
        
        def send_state_messages():
            """Background thread to send state messages every 0.1s until race starts."""
            while not stop_sending.is_set():
                if self.control_mode == "attitude":
                    dummy_action = np.zeros(4, dtype=np.float32) 
                else:
                    dummy_action = np.zeros(13, dtype=np.float32)
                    dummy_action[:3] = current_positions[self.rank]  # Send current position as dummy action
                    dummy_action[2] = 0.3
                self._send_state_update(dummy_action, stopped=False)
                time.sleep(0.1)
        
        sender_thread = threading.Thread(target=send_state_messages, daemon=True)
        sender_thread.start()
        
        # Wait for host ready with timeout
        if not self._host_ready_event.wait(timeout=max_wait_time):
            stop_sending.set()
            raise TimeoutError(
                f"Client {self.rank}: Timeout waiting for host ready after {max_wait_time}s. "
                "Host may not be running or network connection failed."
            )
        
        stop_sending.set()
        logger.info(f"Client {self.rank}: Environment reset complete")
        return self.obs(), self.info()
    
    def lock_until_race_start(self, timeout: float = 60.0):
        """Block until the race starts, with a timeout."""
        logger.info(f"Client {self.rank}: Waiting for race to start...")
        start_time = time.time()
        while not self._race_started:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Client {self.rank}: Timeout waiting for race to start after {timeout}s.")
            
        logger.info(f"Client {self.rank}: Race started, proceeding with control loop")

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a control step.
        
        Args:
            action: Control action for this drone.
            
        Returns:
            Observation, reward, terminated, truncated, info.
        """    
        
        # Get drone states
        drone_pos, _, _, _ = self._get_all_drone_states()
        
        # Check sensor visibility
        dpos = drone_pos[:, None, :2] - self.gates.pos[None, :, :2]
        self.data.gates_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        dpos = drone_pos[:, None, :2] - self.obstacles.pos[None, :, :2]
        self.data.obstacles_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        
        # Check gate passage
        gate_pos = self.gates.pos[self.data.target_gate]
        gate_quat = self.gates.quat[self.data.target_gate]
        
        with jax.default_device(self.device):
            passed = gate_passed(drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
        
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1
        
        # Check if this drone finished or failed
        terminated = self.data.target_gate[self.rank] == -1
        
        # Check safety bounds
        if np.any((self.pos_limit_low > drone_pos[self.rank, :]) | (drone_pos[self.rank, :] > self.pos_limit_high)):
            logger.warning(f"Client {self.rank}: Drone exceeded safety bounds")
            terminated = True

        # Send action to drone (via Zenoh to host) and to ROS for external estimator
        self._send_action_ros(action)
        # Send state message to host
        self._send_state_update(action, terminated)
        
        # Mark stopped if terminated
        if terminated:
            self._should_stop = True
        
        return self.obs(), 0.0, terminated, False, self.info()
    
    def _send_state_update(self, action: NDArray, stopped: bool):
        """Send state update to host.
        
        Args:
            action: Current control action.
            stopped: Whether this client has stopped.
        """
        elapsed_time = time.time() - self._race_start_time if self._race_started else 0.0
        
        # Adjust timestamp to match host's clock using calibrated offset
        adjusted_timestamp = time.time() + self._clock_offset
        
        state_msg = ClientStateMessage(
            drone_rank=self.rank,
            action=action.tolist() if isinstance(action, np.ndarray) else list(action),
            elapsed_time=elapsed_time,
            timestamp=adjusted_timestamp,
            stopped=stopped or self._should_stop,
            next_gate_idx=int(self.data.target_gate[self.rank]),
        )
        self._client_state_pub.publish(state_msg)
    
    def _send_action_ros(self, action: NDArray):
        """Publishes to ROS for external estimator to track commands.
        Args:
            action: Control action.
        """
        # Publish command to ROS for external estimator
        if self.control_mode == "attitude":
            if self._ros_connector_own is not None:
                self._ros_connector_own.publish_cmd(action)
            else:
                logger.warning(f"Client {self.rank}: ROS connector not initialized, cannot send command!")

    
    def obs(self) -> dict[str, NDArray]:
        """Get current observation."""
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
        """Get info dictionary."""
        return {'rank': self.rank}
    
    def close(self):
        """Close the environment."""
        logger.info(f"Client {self.rank}: Closing environment...")
        
        # Send final stop message
        if self._client_state_pub:
            if self.control_mode == "attitude":
                self._send_state_update(np.zeros(4), stopped=True)
        
        # Close Zenoh subscribers and publishers
        if self._host_ready_sub:
            self._host_ready_sub.close()
        if self._host_initialized_sub:
            self._host_initialized_sub.close()
        if self._host_pong_sub:
            self._host_pong_sub.close()
        if self._race_start_sub:
            self._race_start_sub.close()
        if self._client_state_pub:
            self._client_state_pub.close()
        if self._client_ping_pub:
            self._client_ping_pub.close()
        if self._zenoh_session:
            self._zenoh_session.close()
        
        logger.info(f"Client {self.rank}: Environment closed")

