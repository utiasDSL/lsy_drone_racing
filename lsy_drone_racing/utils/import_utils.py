"""Helper module to load the crazyswarm module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import crazyswarm  # noqa: F401
import rospkg

logger = logging.getLogger(__name__)


def get_ros_package_path(pkg: str, heuristic_search: bool = False) -> Path:
    """Get the path to a ROS package.

    If the package is not found and heuristic_search is enabled, we search for the package manually
    in the user's home directory. Any directory with the pattern *_ws is considered a workspace. We
    then check if the crazyswarm folder is present in the src directory of the workspace.

    Args:
        pkg: The name of the ROS package.
        heuristic_search: Flag to enable search heuristics if ROS cannot find the package.

    Returns:
        The path to the ROS package.
    """
    try:
        return Path(rospkg.RosPack().get_path(pkg))
    except rospkg.common.ResourceNotFound as e:
        if not heuristic_search:
            raise e
    logger.warning(f"ROS package {pkg} not found. Searching for the package manually.")
    home = Path.home()
    for path in (d for d in home.glob("*_ws") if d.is_dir()):
        if not (path / f"src/{pkg}").is_dir():
            continue
        pkg_path = path / f"src/{pkg}"
        if not pkg_path.is_dir():
            continue
        return pkg_path
    raise ModuleNotFoundError(f"ROS package {pkg} not found.")


try:
    import pycrazyswarm  # noqa: F401
except ImportError:
    path = get_ros_package_path("crazyswarm", heuristic_search=True)
    pycrazyswarm_path = path / "scripts"
    if str(pycrazyswarm_path) not in sys.path:
        sys.path.insert(0, str(pycrazyswarm_path))

    import pycrazyswarm  # noqa: F401
