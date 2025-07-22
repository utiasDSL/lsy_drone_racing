"""Module for visualizing objects inside an environment."""

from typing import Callable

import numpy as np

from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import draw_cylinder, draw_ellipsoid, draw_line


class Visualizer:
    """Class for visualizing objects in an environment."""

    def __init__(self, env: DroneRaceEnv):
        """Initialize the Visualizer with the environment.

        Args:
            env (DroneRaceEnv): The environment to add objects to visualize.
        """
        self.env = env
        self.line_visualizations = []
        self.cylinder_visualizations = []
        self.ellipsoid_visualizations = []

    def register_line_visualization(
        self, get_line_function: Callable, rgba: np.ndarray = np.array([0.0, 0.0, 1.0, 1.0])
    ):
        """Add a line visualization to the environment.

        Args:
            get_line_function (Callable): Function that returns the line data to visualize.
            rgba (np.ndarray, optional): Color values for the visualization. Defaults to np.array([0.0, 0.0, 1.0, 1.0]).
        """
        self.line_visualizations.append({"function": get_line_function, "rgba": rgba})

    def register_cylinder_visualizations(
        self, get_cylinder_function: Callable, rgba: np.ndarray = np.array([0.0, 1.0, 0.0, 1.0])
    ):
        """Add a cylinder visualization to the environment.

        Args:
            get_cylinder_function (Callable): Function that returns the cylinder parameters to visualize.
            rgba (np.ndarray, optional): Color values for the visualization. Defaults to np.array([0.0, 1.0, 0.0, 1.0]).
        """
        self.cylinder_visualizations.append({"function": get_cylinder_function, "rgba": rgba})

    def register_ellipsoid_visualizations(
        self, get_ellipsoid_function: Callable, rgba: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0])
    ):
        """Add an ellipsoid visualization to the environment.

        Args:
            get_ellipsoid_function (Callable): Function that returns the ellipsoid parameters to visualize.
            rgba (np.ndarray, optional): Color values for the visualization. Defaults to np.array([1.0, 0.0, 0.0, 1.0]).
        """
        self.ellipsoid_visualizations.append({"function": get_ellipsoid_function, "rgba": rgba})

    def visualize(self):
        """Visualize the environment using the registered visualizations."""
        # Visualize all registered lines
        for line_visualization in self.line_visualizations:
            line_data = line_visualization["function"]()
            if line_data is not None:
                draw_line(
                    self.env, line_data, line_visualization["rgba"], min_size=0.01, max_size=0.01
                )

        # Visualize all registered cylinders
        for cylinder_visualization in self.cylinder_visualizations:
            cylinder_params = cylinder_visualization["function"]()
            for pos, radius in zip(cylinder_params["pos"], cylinder_params["radius"]):
                draw_cylinder(
                    self.env,
                    pos=pos,
                    size=np.array([radius, 2.0]),
                    rgba=cylinder_visualization["rgba"],
                )

        # Visualize all registered ellipsoids
        for ellipsoid_visualization in self.ellipsoid_visualizations:
            ellipsoid_params = ellipsoid_visualization["function"]()
            for pos, axes, rot in zip(
                ellipsoid_params["pos"], ellipsoid_params["axes"], ellipsoid_params["rot"]
            ):
                draw_ellipsoid(
                    self.env, pos=pos, size=axes, rot=rot, rgba=ellipsoid_visualization["rgba"]
                )
