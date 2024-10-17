"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* :class:`~.BaseController`: The abstract base class defining the interface for all controllers.
* :class:`PPOController <lsy_drone_racing.control.ppo_controller.PPOController>`: An example
  implementation using a pre-trained Proximal Policy Optimization (PPO) model.
* :class:`PPOController <lsy_drone_racing.control.trajectory_controller.TrajectoryController>`: A
  controller that follows a pre-defined trajectory using cubic spline interpolation.
"""

from lsy_drone_racing.control.controller import BaseController

__all__ = ["BaseController"]
