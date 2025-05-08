"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* :class:`~.Controller`: The abstract base class defining the interface for all controllers.
* :class:`TrajectoryController <lsy_drone_racing.control.trajectory_controller.TrajectoryController>`:
  A controller that follows a pre-defined trajectory using cubic spline interpolation.
* :class:`AttitudeController <lsy_drone_racing.control.attitude_controller.AttitudeController>`: A
  controller that follows a pre-defined attitude using cubic spline interpolation.
"""

from lsy_drone_racing.control.controller import Controller

__all__ = ["Controller"]
