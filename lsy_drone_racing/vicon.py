from __future__ import annotations

import numpy as np
import rospy
import yaml
from rosgraph import Master
from tf2_msgs.msg import TFMessage

from lsy_drone_racing.import_utils import get_ros_package_path
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
        assert Master("/rosnode").is_online(), "ROS is not running. Please run hover.launch first!"
        try:
            rospy.init_node("playback_node")
        except rospy.exceptions.ROSException:
            ...  # ROS node is already running which is fine for us
        config_path = get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml"
        assert config_path.exists(), "Crazyfly config file missing!"
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        assert len(config["crazyflies"]) == 1, "Only one crazyfly allowed at a time!"
        self.drone_name = f"cf{config['crazyflies'][0]['id']}"

        # Register the Vicon subscribers for the drone and any other tracked object
        self.pos: dict[str, np.ndarray] = {"cf": np.array([])}
        self.rpy: dict[str, np.ndarray] = {"cf": np.array([])}
        for track_name in track_names:  # Initialize the objects' pose
            self.pos[track_name], self.rpy[track_name] = np.array([]), np.array([])

        self.sub = rospy.Subscriber("/tf", TFMessage, self.save_pose)

    def save_pose(self, data: TFMessage):
        """Save the position and orientation of all transforms.

        Args:
            data: The TF message containing the objects' pose.
        """
        for tf in data.transforms:
            name = "cf" if tf.child_frame_id == self.drone_name else tf.child_frame_id
            T, R = tf.transform.translation, tf.transform.rotation
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
        return all(p.size > 0 for p in self.pos.values())
