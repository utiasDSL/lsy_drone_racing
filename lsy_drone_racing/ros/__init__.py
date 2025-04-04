"""ROS functionalities for the LSY Drone Racing project.

This package provides interfaces for communication between the drone racing framework and ROS2
(Robot Operating System). It includes:

* :class:`~lsy_drone_racing.ros.ros_connector.ROSConnector`: A non-blocking interface for ROS2
  communication using multiprocessing, providing efficient access to pose, velocity, and other state
  data from ROS topics.

* Utility functions in :mod:`~lsy_drone_racing.ros.ros_utils` for validating race track setup and
  drone positioning.

The ROS integration is designed for real-world deployment of drone racing algorithms, enabling
communication with motion capture systems (like Vicon) and physical drones through ROS topics. The
implementation uses multiprocessing to ensure that ROS callbacks don't block the main control loop,
with data shared through synchronized memory arrays.
"""

from lsy_drone_racing.ros.ros_connector import ROSConnector

__all__ = ["ROSConnector"]
