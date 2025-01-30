"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Type

import jax
import toml
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict

from lsy_drone_racing.control.controller import BaseController

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from jax import Array

logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[BaseController]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, BaseController)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, BaseController)]
    assert len(controllers) > 0, (
        f"No controller found in {path}. Have you subclassed BaseController?"
    )
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, BaseController)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


@jax.jit
@partial(vectorize, signature="(3),(3),(3),(4)->()", excluded=[4])
def gate_passed(
    drone_pos: Array,
    last_drone_pos: Array,
    gate_pos: Array,
    gate_quat: Array,
    gate_size: tuple[float, float],
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the y-Axis (pointing from -y to +y). Therefore, we check if y
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the x
    and z box boundaries. First, we linearly interpolate to get the x and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
        gate_pos: The position of the gate in the world frame.
        gate_quat: The rotation of the gate as a wxyz quaternion.
        gate_size: The size of the gate box in meters.
    """
    # Transform last and current drone position into current gate frame.
    gate_rot = R.from_quat(gate_quat)
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    passed_plane = (last_pos_local[1] < 0) & (pos_local[1] > 0)
    alpha = -last_pos_local[1] / (pos_local[1] - last_pos_local[1])
    x_intersect = alpha * (pos_local[0]) + (1 - alpha) * last_pos_local[0]
    z_intersect = alpha * (pos_local[2]) + (1 - alpha) * last_pos_local[2]
    # Divide gate size by 2 to get the distance from the center to the edges
    in_box = (abs(x_intersect) < gate_size[0] / 2) & (abs(z_intersect) < gate_size[1] / 2)
    return passed_plane & in_box
