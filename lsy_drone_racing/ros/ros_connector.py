"""The Vicon module provides an interface to the Vicon motion capture system for position tracking.

It defines the Vicon class, which handles communication with the Vicon system through ROS messages.
The Vicon class is responsible for:

* Tracking the drone and other objects (gates, obstacles) in the racing environment.
* Providing real-time pose (position and orientation) data for tracked objects.
* Calculating velocities and angular velocities based on pose changes.

This module is necessary to provide the real-world positioning data for the drone and race track
elements.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import time
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
    from multiprocessing.synchronize import Event


class ROSConnector:
    """Vicon interface for the pose estimation data for the drone and any other tracked objects.

    Vicon sends a stream of ROS messages containing the current pose data. We subscribe to these
    messages and save the pose data for each object in dictionaries. Users can then retrieve the
    latest pose data directly from these dictionaries.
    """

    def __init__(
        self,
        tf_names: list[str] | None = None,
        estimator_names: list[str] | None = None,
        timeout: float = 1.0,
    ):
        """Load the crazyflies.yaml file and register the subscribers for the Vicon pose data.

        Args:
            tf_names: The name of objects that only require pose tracking from /tf.
            estimator_names: The name of objects that should be tracked with state estimation.
            timeout: If greater than 0, Vicon waits for position updates of all tracked objects
                before returning.
        """
        assert rclpy.ok(), "ROS must be initialized before creating a ROSConnector instance."
        self.tf_names = [] if tf_names is None else tf_names
        self.estimator_names = [] if estimator_names is None else estimator_names
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
        """Get the latest position data for all tracked objects."""
        pos = np.asarray(self._pos, dtype=np.float32).reshape(-1, 3)
        return {n: pos[i] for i, n in enumerate(self.names)}

    @property
    def quat(self) -> dict[str, np.ndarray]:
        """Get the latest orientation data for all tracked objects."""
        quat = np.asarray(self._quat, dtype=np.float32).reshape(-1, 4)
        return {n: quat[i] for i, n in enumerate(self.names)}

    @property
    def vel(self) -> dict[str, np.ndarray]:
        """Get the latest velocity data for all tracked objects."""
        vel = np.asarray(self._vel, dtype=np.float32).reshape(-1, 3)
        return {n: vel[i] for i, n in enumerate(self.estimator_names)}

    @property
    def ang_vel(self) -> dict[str, np.ndarray]:
        """Get the latest angular velocity data for all tracked objects."""
        ang_vel = np.asarray(self._ang_vel, dtype=np.float32).reshape(-1, 3)
        return {n: ang_vel[i] for i, n in enumerate(self.estimator_names)}

    @property
    def disturbance(self) -> dict[str, np.ndarray]:
        """Get the latest disturbance data for all tracked objects."""
        disturbance = np.asarray(self._disturbance, dtype=np.float32).reshape(-1, 6)
        return {n: disturbance[i] for i, n in enumerate(self.estimator_names)}

    @property
    def forces(self) -> dict[str, np.ndarray]:
        """Get the latest estimated forces data for all tracked objects."""
        forces = np.asarray(self._forces, dtype=np.float32).reshape(-1, 6)
        return {n: forces[i] for i, n in enumerate(self.estimator_names)}

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest pose of a tracked object.

        Args:
            name: The name of the object.

        Returns:
            The position and orientation (as xyzw quaternion) of the object.
        """
        return self.pos[name], self.quat[name]

    @property
    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest poses of all objects."""
        return np.stack(self.pos.values()), np.stack(self.quat.values())

    @property
    def active(self) -> bool:
        """Check if Vicon has sent data for each object."""
        # Check if drone is being tracked and if drone has already received updates
        pos_ready = not np.any(np.isnan(np.array(self._pos, dtype=np.float32)))
        quat_ready = not np.any(np.isnan(np.array(self._quat, dtype=np.float32)))
        vel_ready = not np.any(np.isnan(np.array(self._vel, dtype=np.float32)))
        ang_vel_ready = not np.any(np.isnan(np.array(self._ang_vel, dtype=np.float32)))
        return pos_ready and quat_ready and vel_ready and ang_vel_ready

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
        names: The names of the objects to track.
        pos: The shared memory arrays for the position data.
        quat: The shared memory arrays for the orientation data.
        times: The shared memory arrays for the time data.
    """
    if not names:
        return
    rclpy.init()
    node = rclpy.create_node("pose_updater_" + uuid4().hex)

    subs = []
    for name in names:
        # Pose (pos and quat) are shared with all_names, so the index changes!
        fn = partial(pose_callback, idx=all_names.index(name), pos=pos, quat=quat, times=times)
        sub = node.create_subscription(PoseStamped, f"/estimated_state_pose_{name}", fn, 10)
        subs.append(sub)
        fn = partial(twist_callback, idx=names.index(name), vel=vel, ang_vel=ang_vel)
        sub = node.create_subscription(TwistStamped, f"/estimated_state_twist_{name}", fn, 10)
        subs.append(sub)
        fn = partial(disturbance_callback, idx=names.index(name), disturbance=disturbance)
        sub = node.create_subscription(
            WrenchStamped, f"/estimated_state_disturbance_{name}", fn, 10
        )
        subs.append(sub)
        fn = partial(forces_callback, idx=names.index(name), forces=forces)
        sub = node.create_subscription(Float64MultiArray, f"/estimated_state_forces_{name}", fn, 10)
        subs.append(sub)

    while not shutdown.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
    for sub in subs:
        sub.destroy()
    node.destroy_node()


def tf_callback(
    data: TFMessage,
    names: list[str],
    pos: SynchronizedArray,
    quat: SynchronizedArray,
    times: SynchronizedArray,
):
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
    pos: SynchronizedArray,
    quat: SynchronizedArray,
    times: Synchronized,
    idx: int,
):
    """Save the drone pose from the estimator node.

    Args:
        data: The PoseStamped message.
        idx: The index of the object.
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
        name: The name of the object.
    """
    linear, angular = data.twist.linear, data.twist.angular
    vel[idx * 3 : (idx + 1) * 3] = [linear.x, linear.y, linear.z]
    ang_vel[idx * 3 : (idx + 1) * 3] = [angular.x, angular.y, angular.z]


def disturbance_callback(data: WrenchStamped, idx: int, disturbance: SynchronizedArray):
    """Save the disturbance force and torque from the estimator node.

    Args:
        data: The WrenchStamped message.
        name: The name of the object.
    """
    force, torque = data.wrench.force, data.wrench.torque
    disturbance[idx * 6 + (idx + 1) * 6] = [force.x, force.y, force.z, torque.x, torque.y, torque.z]


def forces_callback(data: Float64MultiArray, idx: int, forces: SynchronizedArray):
    """Save the estimated forces from the estimator node.

    Args:
        data: The Float64MultiArray message.
        name: The name of the object
    """
    forces[idx * 4 : (idx + 1) * 4] = data.data
