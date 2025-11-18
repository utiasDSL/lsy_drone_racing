"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type

import mujoco
import numpy as np
import toml
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from lsy_drone_racing.envs.race_core import RaceCoreEnv
    from lsy_drone_racing.envs.real_race_env import RealDroneRaceEnv


logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
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
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

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


def save_track_layout(env: RealDroneRaceEnv, config: ConfigDict, output_path: Path):
    """Save the track objects poses in the real race environment to a toml file.

    Args:
        env: The RealDroneRaceEnv, which stores actual gates, objstacles and drone positions from MoCap
        config: The original ConfigDict to copy
        output_path: Path object to the output toml config file.
        real_pos: Boolean. If set to True, the real poses of track objects are stored. Otherwise the nominal poses are stored.
    
    """
    assert output_path.suffix == ".toml", f"Configuration file has to be a TOML file: {output_path}"
    keys_to_copy = ['controller', 'deploy', 'env', 'sim']
    config_dict = config.to_dict()
    output_dict = {
        key: config_dict[key].to_dict() for key in keys_to_copy
    }

    # Now the track.randomize needs be set to false,
    # since we are storing a layout from real-world placement
    output_dict['env']['track']['randomize'] = False
    # Overwrite the original field with the actual positions from the env
    output_dict['env']['track']['gates'] = [
        {'pos' : (env.unwrapped.gates.pos[i]).tolist(),
        'rpy' : (R.from_quat(env.unwrapped.gates.quat[i]).as_euler('xyz', degrees = False)).tolist()
        }
        for i in range(env.unwrapped.gates.pos.shape[0])
    ]
    output_dict['env']['track']['obstacles'] = [
        {'pos' : (env.unwrapped.obstacles.pos[i]).tolist(),
        }
        for i in range(env.unwrapped.obstacles.pos.shape[0])
    ]
    
    with open(output_path, "w") as f:
        toml.dump(output_dict, f)

def draw_line(
    env: RaceCoreEnv,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        env: The drone racing environment.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    sim = env.unwrapped.sim
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
