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

import time

import numpy as np
import rospy
import yaml
from crazyswarm.msg import StateVector
from rosgraph import Master
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage

from lsy_drone_racing.utils.import_utils import get_ros_package_path


class Vicon:
    """Vicon interface for the pose estimation data for the drone and any other tracked objects.

    Vicon sends a stream of ROS messages containing the current pose data. We subscribe to these
    messages and save the pose data for each object in dictionaries. Users can then retrieve the
    latest pose data directly from these dictionaries.
    """

    def __init__(
        self, track_names: list[str] = [], auto_track_drone: bool = True, timeout: float = 0.0
    ):
        """Load the crazyflies.yaml file and register the subscribers for the Vicon pose data.

        Args:
            track_names: The names of any additional objects besides the drone to track.
            auto_track_drone: Infer the drone name and add it to the positions if True.
            timeout: If greater than 0, Vicon waits for position updates of all tracked objects
                before returning.
        """
        assert Master("/rosnode").is_online(), "ROS is not running. Please run hover.launch first!"
        try:
            rospy.init_node("playback_node")
        except rospy.exceptions.ROSException:
            ...  # ROS node is already running which is fine for us
        self.drone_name = None
        self.auto_track_drone = auto_track_drone
        if auto_track_drone:
            with open(get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml", "r") as f:
                config = yaml.load(f, yaml.SafeLoader)
            assert len(config["crazyflies"]) == 1, "Only one crazyfly allowed at a time!"
            self.drone_name = f"cf{config['crazyflies'][0]['id']}"
        self.track_names = track_names
        # Register the Vicon subscribers for the drone and any other tracked object
        self.pos: dict[str, np.ndarray] = {}
        self.rpy: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.ang_vel: dict[str, np.ndarray] = {}
        self.time: dict[str, float] = {}

        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        if auto_track_drone:
            self.estimator_sub = rospy.Subscriber(
                "/estimated_state", StateVector, self.estimator_callback
            )

        if timeout:
            tstart = time.time()
            while not self.active and time.time() - tstart < timeout:
                time.sleep(0.01)
            if not self.active:
                raise TimeoutError(
                    "Timeout while fetching initial position updates for all tracked objects. "
                    f"Missing objects: {[k for k in self.track_names if k not in self.ang_vel]}"
                )

    def estimator_callback(self, data: StateVector):
        """Save the drone state from the estimator node.

        Args:
            data: The StateVector message.
        """
        if self.drone_name is None:
            return
        self.pos[self.drone_name] = np.array(data.pos)
        rpy = R.from_quat(data.quat).as_euler("xyz")
        self.rpy[self.drone_name] = np.array(rpy)
        self.vel[self.drone_name] = np.array(data.vel)
        self.ang_vel[self.drone_name] = np.array(data.omega_b)

    def tf_callback(self, data: TFMessage):
        """Save the position and orientation of all transforms.

        Args:
            data: The TF message containing the objects' pose.
        """
        for tf in data.transforms:
            name = tf.child_frame_id.split("/")[-1]
            # Skip drone if it is also in track names, handled by the estimator_callback
            if name == self.drone_name:
                continue
            if name not in self.track_names:
                continue
            T, Rot = tf.transform.translation, tf.transform.rotation
            pos = np.array([T.x, T.y, T.z])
            rpy = R.from_quat([Rot.x, Rot.y, Rot.z, Rot.w]).as_euler("xyz")
            self.time[name] = time.time()
            self.pos[name] = pos
            self.rpy[name] = rpy

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest pose of a tracked object.

        Args:
            name: The name of the object.

        Returns:
            The position and rotation of the object. The rotation is in roll-pitch-yaw format.
        """
        return self.pos[name], self.rpy[name]

    @property
    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest poses of all objects."""
        return np.stack(self.pos.values()), np.stack(self.rpy.values())

    @property
    def names(self) -> list[str]:
        """Get a list of actively tracked names."""
        return list(self.pos.keys())

    @property
    def active(self) -> bool:
        """Check if Vicon has sent data for each object."""
        # Check if drone is being tracked and if drone has already received updates
        if self.drone_name is not None and self.drone_name not in self.pos:
            return False
        # Check remaining object's update status
        return all([name in self.pos for name in self.track_names])

    def close(self):
        """Unregister the ROS subscribers."""
        self.tf_sub.unregister()
        if self.auto_track_drone:
            self.estimator_sub.unregister()
