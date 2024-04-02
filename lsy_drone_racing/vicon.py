from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import TransformStamped

from lsy_drone_racing.utils import euler_from_quaternion


class ViconWatcher:
    """Vicon interface for the pose estimation data for the drone and any other tracked objects.

    Vicon sends a stream of ROS messages containing the current pose data. We subscribe to these
    messages and save the pose data for each object in dictionaries. Users can then retrieve the
    latest pose data directly from these dictionaries.
    """

    def __init__(self, track_names: list[str] = []):
        """Load the crazyflies.yaml file and register the subscribers for the Vicon pose data.

        Args:
            track_names: The names of any additional objects besides the drone to track.
        """
        # rospy.init_node("playback_node")
        config_path = Path(__file__).resolve().parents[2] / "launch/crazyflies.yaml"
        assert config_path.exists(), "Crazyfly config file missing!"
        with open(config_path, "r") as f:
            config = yaml.load(f)
        assert len(config["crazyflies"]) == 1, "Only one crazyfly allowed at a time!"

        # Register the Vicon subscribers for the drone and any other tracked object
        self.subs = {}
        self.pos = {"cf": None}
        self.rpy = {"cf": None}
        cf_id = "cf" + str(config["crazyflies"][0]["id"])
        msg_topic = f"/vicon/{cf_id}/{cf_id}"
        callback = partial(self.save_pose, name="cf")
        self.subs["cf"] = rospy.Subscriber(msg_topic, TransformStamped, callback)
        # Register the Vicon subscribers for the other tracked objects
        for track_name in track_names:
            self.pos[track_name], self.rpy[track_name] = None, None  # Initialize the object's pose
            msg_topic = f"/vicon/{track_name}/{track_name}"
            callback = partial(self.save_pose, name=track_name)
            self.subs[track_name] = rospy.Subscriber(msg_topic, TransformStamped, callback)

    def save_pose(self, data: TransformStamped, name: str):
        """Save the position and orientation of the object.

        Args:
            data: The ROS message containing the object's pose.
            name: The name of the object.
        """
        T, R = data.transform.translation, data.transform.rotation
        self.pos[name] = np.array([T.x, T.y, T.z])
        self.rpy[name] = np.array(euler_from_quaternion(R.x, R.y, R.z, R.w))

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest pose of a tracked object.

        Args:
            name: The name of the object.

        Returns:
            The position and rotation of the object. The rotation is in roll-pitch-yaw format.
        """
        return self.pos[name], self.rpy[name]

    @property
    def active(self) -> bool:
        """Check if Vicon has sent data for each object."""
        return all(p is not None for p in self.pos.values())
