"""The ROS connector module provides an interface for communication with ROS2 topics.

It defines the ROSConnector class, which handles communication with ROS2 through multiprocessing
to ensure non-blocking operation. The module is responsible for:

* Providing real-time state estimation (position, orientation, velocity, ...) for drones
* Tracking objects (gates, obstacles) through ROS /tf
* Publishing commands to ROS topics asynchronously for estimator nodes and logging

The main objective for the ROSConnector is to interface with ROS2 messages with minimal latency. It
uses a multiprocessing architecture to prevent ROS callbacks from blocking the main application
thread. Data is shared between processes using synchronized shared memory arrays, allowing for
efficient, thread-safe data access without copying.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import time
from functools import partial
from queue import Empty
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
    from multiprocessing.synchronize import Event

    from numpy.typing import NDArray


class ROSConnector:
    """A non-blocking interface for ROS2 communication using multiprocessing.

    The ROSConnector class provides a thread-safe way to receive data from ROS2 topics and publish
    commands without blocking the main application thread. It uses separate processes for
    subscribing to topics and publishing commands, with data shared through synchronized memory
    arrays.

    The class supports two types of tracking:

    * Simple pose tracking from `/tf` topics
    * Full state estimation including velocity, angular velocity, and disturbance forces from an
      estimator node.

    Data is accessed through property methods that convert the shared memory arrays into
    dictionaries keyed by object names.

    Example:
        >>> # Track drone position and state from estimator
        >>> conn = ROSConnector(
        ...     estimator_names=["cf1"], cmd_topic="/estimator/cf1/cmd", timeout=1.0
        ... )
        >>> # Access the latest position data
        >>> drone_pos = conn.pos["cf1"]
        >>> # Publish a command (non-blocking)
        >>> conn.publish_cmd(np.array([1.0, 0.0, 0.0, 0.0]))
        >>> # Clean up when done
        >>> conn.close()
    """

    def __init__(
        self,
        tf_names: list[str] | None = None,
        estimator_names: list[str] | None = None,
        cmd_topic: str | None = None,
        timeout: float = 1.0,
    ):
        """Initialize the ROSConnector with the specified tracking objects and command topic.

        Note:
            It is strongly recommended to use the timeout parameter to ensure that the connector has
            valid data for all objects. Otherwise, data may contain NaN values until the first
            update has been received.

        Args:
            tf_names: Names of objects to track using only pose data from `/tf` topic. If None,
                nothing will be tracked.
            estimator_names: Names of objects to track with full state estimation. If None, nothing
                will be tracked.
            cmd_topic: Topic name for publishing commands. If None, command publishing is disabled.
            timeout: If greater than 0, wait for position updates of all tracked objects before
                returning.

        Raises:
            AssertionError: If ROS is not initialized before creating the connector.
            TimeoutError: If updates for all tracked objects aren't received within the specified
                timeout period.
        """
        assert rclpy.ok(), "ROS must be initialized before creating a ROSConnector instance."
        self.tf_names = [] if tf_names is None else tf_names
        self.estimator_names = [] if estimator_names is None else estimator_names
        assert not set(self.tf_names).intersection(set(self.estimator_names)), (
            "Duplicate names in tf and estimator"
        )
        assert len(self.tf_names) == len(set(self.tf_names)), "Duplicate items in tf_names"
        assert len(self.estimator_names) == len(set(self.estimator_names)), (
            "Duplicate items in estimator_names"
        )
        self.names = self.tf_names + self.estimator_names
        # Create synchronized, shared arrays for all quantities
        ctx = mp.get_context("spawn")
        self._pos = ctx.Array("f", [float("nan")] * len(self.names) * 3)
        self._quat = ctx.Array("f", [float("nan")] * len(self.names) * 4)
        self._vel = ctx.Array("f", [float("nan")] * len(self.estimator_names) * 3)
        self._ang_vel = ctx.Array("f", [float("nan")] * len(self.estimator_names) * 3)
        self._disturbance = ctx.Array("f", [float("nan")] * len(self.estimator_names) * 6)
        self._forces = ctx.Array("f", [float("nan")] * len(self.estimator_names) * 4)
        self._times = ctx.Array("f", [float("nan")] * len(self.names))

        self.shutdown = ctx.Event()
        atexit.register(lambda: self.shutdown.set())  # Ensure processes are killed on exit
        self.processes = []
        if self.tf_names:  # Create a process for the /tf callback
            tf_process = ctx.Process(
                target=tf_update,
                args=(self.tf_names, self._pos, self._quat, self._times, self.shutdown),
            )
            self.processes.append(tf_process)

        if self.estimator_names:  # Create a process for each estimator callback
            estimator_process = ctx.Process(
                target=estimate_update,
                args=(
                    self.names,
                    self.estimator_names,
                    self._pos,
                    self._quat,
                    self._vel,
                    self._ang_vel,
                    self._disturbance,
                    self._forces,
                    self._times,
                    self.shutdown,
                ),
            )
            self.processes.append(estimator_process)

        self.cmd_pub, self.cmd_queue = None, None
        if cmd_topic is not None:  # Create a process for the command publisher
            self.cmd_queue = ctx.Queue(maxsize=10)
            self.cmd_pub = ctx.Process(
                target=command_publisher, args=(self.cmd_queue, cmd_topic, self.shutdown)
            )
            self.cmd_pub.start()

        for p in self.processes:
            p.start()

        # Timeouts are highly recommended to avoid reading NaNs and crashing while processes are
        # still spinning up
        if timeout:
            tstart = time.time()
            while time.time() - tstart < timeout:
                if self.active:
                    break
                time.sleep(0.01)  # Processes are spinning, so we can sleep here
            else:
                self.shutdown.set()
                missing_objects = [name for name, pos in self.pos.items() if np.any(np.isnan(pos))]
                raise TimeoutError(
                    "Timeout while fetching initial position updates for all tracked objects. "
                    f"Missing objects: {missing_objects}"
                )

    @property
    def pos(self) -> dict[str, np.ndarray]:
        """The latest position data for all tracked objects."""
        pos = np.asarray(self._pos, dtype=np.float32).reshape(-1, 3)
        return {n: pos[i] for i, n in enumerate(self.names)}

    @property
    def quat(self) -> dict[str, np.ndarray]:
        """The latest orientation data for all tracked objects."""
        quat = np.asarray(self._quat, dtype=np.float32).reshape(-1, 4)
        return {n: quat[i] for i, n in enumerate(self.names)}

    @property
    def vel(self) -> dict[str, np.ndarray]:
        """The latest velocity data for all tracked objects."""
        vel = np.asarray(self._vel, dtype=np.float32).reshape(-1, 3)
        return {n: vel[i] for i, n in enumerate(self.estimator_names)}

    @property
    def ang_vel(self) -> dict[str, np.ndarray]:
        """The latest angular velocity data for all tracked objects."""
        ang_vel = np.asarray(self._ang_vel, dtype=np.float32).reshape(-1, 3)
        return {n: ang_vel[i] for i, n in enumerate(self.estimator_names)}

    @property
    def disturbance(self) -> dict[str, np.ndarray]:
        """The latest disturbance data for all tracked objects."""
        disturbance = np.asarray(self._disturbance, dtype=np.float32).reshape(-1, 6)
        return {n: disturbance[i] for i, n in enumerate(self.estimator_names)}

    @property
    def forces(self) -> dict[str, np.ndarray]:
        """The latest estimated forces data for all tracked objects."""
        forces = np.asarray(self._forces, dtype=np.float32).reshape(-1, 6)
        return {n: forces[i] for i, n in enumerate(self.estimator_names)}

    @property
    def active(self) -> bool:
        """Check if each object has received an update already."""
        # Check if drone is being tracked and if drone has already received updates
        pos_ready = not np.any(np.isnan(np.array(self._pos, dtype=np.float32)))
        quat_ready = not np.any(np.isnan(np.array(self._quat, dtype=np.float32)))
        vel_ready = not np.any(np.isnan(np.array(self._vel, dtype=np.float32)))
        ang_vel_ready = not np.any(np.isnan(np.array(self._ang_vel, dtype=np.float32)))
        return pos_ready and quat_ready and vel_ready and ang_vel_ready

    def publish_cmd(self, value: NDArray):
        """Publish a command to ROS.

        This is a non-blocking function that puts the command into a multiprocessing queue. Commands
        are picked up by the command publisher process and published to the ROS topic
        asynchronously.

        Note:
            The command publisher topic must be set on initialization. Otherwise this function will
            raise an error.
        """
        if self.cmd_pub is None:
            raise ValueError("Command publisher not initialized.")
        self.cmd_queue.put(value)

    def close(self):
        """Unregister the ROS subscriptions and shut down the node."""
        self.shutdown.set()
        for p in self.processes:
            p.join()


def tf_update(
    names: list[str],
    pos: list[SynchronizedArray],
    quat: list[SynchronizedArray],
    times: list[Synchronized],
    shutdown: Event,
):
    """Update the shared memory arrays with the latest position and orientation data from /tf.

    Args:
        names: The names of the objects to track.
        pos: The shared memory arrays for the position data.
        quat: The shared memory arrays for the orientation data.
        times: The shared memory arrays for the time data.
        shutdown: The event to signal shutdown.
    """
    rclpy.init()
    node = rclpy.create_node("tf_updater_" + uuid4().hex)
    fn = partial(tf_callback, names=names, pos=pos, quat=quat, times=times)
    sub = node.create_subscription(TFMessage, "/tf", fn, 10)
    while not shutdown.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
    sub.destroy()
    node.destroy_node()


def estimate_update(
    all_names: list[str],
    names: list[str],
    pos: list[SynchronizedArray],
    quat: list[SynchronizedArray],
    vel: list[SynchronizedArray],
    ang_vel: list[SynchronizedArray],
    disturbance: list[SynchronizedArray],
    forces: list[SynchronizedArray],
    times: list[Synchronized],
    shutdown: Event,
):
    """Update the shared memory arrays with the latest pose data from the estimator node.

    Args:
        all_names: The names of all objects that are being tracked.
        names: The names of the objects only being tracked by the estimator.
        pos: The shared memory arrays for the position data.
        quat: The shared memory arrays for the orientation data.
        vel: The shared memory arrays for the velocity data.
        ang_vel: The shared memory arrays for the angular velocity data.
        disturbance: The shared memory arrays for the disturbance data.
        forces: The shared memory arrays for the forces data.
        times: The shared memory arrays for the time data.
        shutdown: The event to signal shutdown.
    """
    rclpy.init()
    node = rclpy.create_node("estimate_updater_" + uuid4().hex)

    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1
    )

    subs = []
    for name in names:
        # Pose (pos and quat) are shared with all_names, so the index changes!
        fn = partial(pose_callback, idx=all_names.index(name), pos=pos, quat=quat, times=times)
        sub = node.create_subscription(
            PoseStamped, f"/drones/{name}/estimate/pose", fn, qos_profile
        )
        subs.append(sub)
        fn = partial(twist_callback, idx=names.index(name), vel=vel, ang_vel=ang_vel)
        sub = node.create_subscription(
            TwistStamped, f"/drones/{name}/estimate/twist", fn, qos_profile
        )
        subs.append(sub)
        fn = partial(disturbance_callback, idx=names.index(name), disturbance=disturbance)
        sub = node.create_subscription(
            WrenchStamped, f"/drones/{name}/estimate/wrench", fn, qos_profile
        )
        subs.append(sub)
        fn = partial(forces_callback, idx=names.index(name), forces=forces)
        sub = node.create_subscription(
            Float64MultiArray, f"/drones/{name}/estimate/forces", fn, qos_profile
        )
        subs.append(sub)

    while not shutdown.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
    for sub in subs:
        sub.destroy()
    node.destroy_node()


def command_publisher(queue: mp.Queue, cmd_topic: str, shutdown: Event):
    """Publish commands from a queue to a ROS topic.

    Args:
        queue: The queue containing commands to publish.
        cmd_topic: The topic name.
        shutdown: The event to signal shutdown.
    """
    rclpy.init()
    node = rclpy.create_node("cmd_publisher_" + uuid4().hex)
    publisher = node.create_publisher(Float64MultiArray, cmd_topic, 10)

    while not shutdown.is_set():
        try:
            cmd = None
            cmd = queue.get(timeout=0.1)  # Get the latest message from the queue. Block if empty
            while True:  # Get the latest message if queue holds more
                cmd = queue.get_nowait()
        except Empty:  # We have cleared the queue or timed out, either is fine
            pass
        if cmd is not None:
            cmd = np.array(cmd, dtype=np.float64)
            publisher.publish(Float64MultiArray(data=cmd))

    node.destroy_node()


def tf_callback(
    data: TFMessage,
    names: list[str],
    pos: SynchronizedArray,
    quat: SynchronizedArray,
    times: SynchronizedArray,
):
    """Save the pose data from the /tf topic into shared memory arrays.

    Args:
        data: The TFMessage message.
        names: The names of the objects to track.
        pos: The shared memory array for the position values.
        quat: The shared memory array for the orientation values.
        times: The shared memory array for the time values.
    """
    for tf in data.transforms:
        name = tf.child_frame_id.split("/")[-1]
        if name in names:
            T, Rot = tf.transform.translation, tf.transform.rotation
            idx = names.index(name)
            pos[idx * 3 : (idx + 1) * 3] = [T.x, T.y, T.z]
            quat[idx * 4 : (idx + 1) * 4] = [Rot.x, Rot.y, Rot.z, Rot.w]
            times[idx] = time.time()


def pose_callback(
    data: PoseStamped,
    idx: int,
    pos: SynchronizedArray,
    quat: SynchronizedArray,
    times: Synchronized,
):
    """Save the drone pose from the estimator node.

    Args:
        data: The PoseStamped message.
        idx: The index of the object the pose is referring to.
        pos: The shared memory array for the position values.
        quat: The shared memory array for the orientation values.
        times: The shared memory array for the time values.
    """
    T, Rot = data.pose.position, data.pose.orientation
    pos[idx * 3 : (idx + 1) * 3] = [T.x, T.y, T.z]
    quat[idx * 4 : (idx + 1) * 4] = [Rot.x, Rot.y, Rot.z, Rot.w]
    times[idx] = time.time()


def twist_callback(
    data: TwistStamped, idx: int, vel: SynchronizedArray, ang_vel: SynchronizedArray
):
    """Save the drone twist (velocity and angular velocity) from the estimator node.

    Args:
        data: The TwistStamped message.
        idx: The index of the object the velocity is referring to.
        vel: The shared memory array for the velocity values.
        ang_vel: The shared memory array for the angular velocity values.
    """
    linear, angular = data.twist.linear, data.twist.angular
    vel[idx * 3 : (idx + 1) * 3] = [linear.x, linear.y, linear.z]
    ang_vel[idx * 3 : (idx + 1) * 3] = [angular.x, angular.y, angular.z]


def disturbance_callback(data: WrenchStamped, idx: int, disturbance: SynchronizedArray):
    """Save the disturbance force and torque from the estimator node.

    Args:
        data: The WrenchStamped message.
        idx: The index of the object the force is referring to.
        disturbance: The shared memory array for the disturbances.
    """
    force, torque = data.wrench.force, data.wrench.torque
    disturbance[idx * 6 : (idx + 1) * 6] = [force.x, force.y, force.z, torque.x, torque.y, torque.z]


def forces_callback(data: Float64MultiArray, idx: int, forces: SynchronizedArray):
    """Save the estimated forces from the estimator node.

    Args:
        data: The Float64MultiArray message.
        idx: The index of the object the force is referring to.
        forces: The shared memory array for the forces.
    """
    forces[idx * 4 : (idx + 1) * 4] = data.data
