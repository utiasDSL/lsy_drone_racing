"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* :class:`~.Controller`: The abstract base class defining the interface for all controllers.
* :class:`TrajectoryController <ece484_fly.control.trajectory_controller.TrajectoryController>`:
  A controller that follows a pre-defined trajectory using cubic spline interpolation.
* :class:`AttitudeController <ece484_fly.control.attitude_controller.AttitudeController>`: A
  controller that follows a pre-defined attitude using cubic spline interpolation.
"""  # noqa: E501, required for linking in the docs

from ece484_fly.control.controller import Controller

__all__ = ["Controller"]
