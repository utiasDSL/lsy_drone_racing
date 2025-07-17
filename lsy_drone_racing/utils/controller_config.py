"""Controller Config Module.

This module provides easy access to controller constants defined in the configuration file.
It loads the constants from the TOML file and provides convenient access methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import toml


class ControllerConfig:
    """Class to manage controller configuration from configuration file."""

    def __init__(self, config_file: str = "controller_config.toml"):
        """Initialize the controller constants.

        Args:
            config_file: Name of the configuration file (default: "controller_config.toml")
        """
        self.config_file = config_file
        # Look for the config file in the config directory (two levels up from utils)
        self.config_path = Path(__file__).parent.parent.parent / "config" / config_file
        self._constants = self._load_constants()

    def _load_constants(self) -> Dict[str, Any]:
        """Load constants from the TOML configuration file.

        Returns:
            Dictionary containing all constants
        """
        try:
            with open(self.config_path, "r") as f:
                return toml.load(f)
        except FileNotFoundError:
            print(f"Warning: Constants file {self.config_path} not found. Using default values.")
            return self._get_default_constants()
        except Exception as e:
            print(f"Error loading constants: {e}. Using default values.")
            return self._get_default_constants()

    def _get_default_constants(self) -> Dict[str, Any]:
        """Get default constants if config file is not available.

        Returns:
            Dictionary with default constants
        """
        return {
            "collision_avoidance": {
                "obstacle_radius": 0.14,
                "gate_length": 0.50,
                "ellipsoid_radius": 0.12,
                "ellipsoid_length": 0.7,
                "ignored_obstacle_indices": [2],
            },
            "mpc": {
                "N": 60,
                "T_HORIZON": 2.0,
                "replanning_frequency": 1,
                "weight_adjustment_duration_ratio": 0.7,
                "weights": {
                    "Q_pos": 8,
                    "Q_vel": 0.01,
                    "Q_rpy": 0.01,
                    "Q_thrust": 0.01,
                    "Q_cmd": 0.01,
                    "R": 0.01,
                },
                "replanning_weights": {"Q_pos_multiplier": 2.0},
                "control": {
                    "last_f_collective_default": 0.3,
                    "last_f_cmd_default": 0.3,
                    "vel_scale_replanning": 1.2,
                    "vel_scale_normal": 1.0,
                    "max_velocity_conservative": 2.0,
                },
            },
            "trajectory_planner": {
                "N_default": 30,
                "T_HORIZON_default": 1.5,
                "approach_dist": [0.2, 0.3, 0.2, 0.1],
                "exit_dist": [0.4, 0.15, 0.2, 5.0],
                "default_approach_dist": 0.1,
                "default_exit_dist": 0.5,
                "approach_height_offset": [0.01, 0.1, 0.0, 0.0],
                "exit_height_offset": [0.1, 0.0, 0.05, 0.0],
                "default_approach_height_offset": 0.1,
                "default_exit_height_offset": 0.0,
            },
            "physics": {"gravity": 9.81},
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a constant value using dot notation.

        Args:
            key_path: Path to the constant using dot notation (e.g., "mpc.weights.Q_pos")
            default: Default value if key is not found

        Returns:
            The constant value or default if not found
        """
        keys = key_path.split(".")
        value = self._constants

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_mpc_weights(self) -> Dict[str, float]:
        """Get MPC weight constants.

        Returns:
            Dictionary with MPC weights
        """
        return self.get("mpc.weights", {})

    def get_replanning_weights(self) -> Dict[str, float]:
        """Get replanning weight constants.

        Returns:
            Dictionary with replanning weights
        """
        base_weights = self.get_mpc_weights()
        multiplier = self.get("mpc.replanning_weights.Q_pos_multiplier", 2.0)

        replanning_weights = base_weights.copy()
        replanning_weights["Q_pos"] *= multiplier

        return replanning_weights

    def get_collision_avoidance_params(self) -> Dict[str, Any]:
        """Get collision avoidance parameters.

        Returns:
            Dictionary with collision avoidance parameters
        """
        return self.get("collision_avoidance", {})

    def get_trajectory_planner_params(self) -> Dict[str, Any]:
        """Get trajectory planner parameters.

        Returns:
            Dictionary with trajectory planner parameters
        """
        return self.get("trajectory_planner", {})

    def get_gate_distances(self) -> Dict[str, Any]:
        """Get gate distance parameters.

        Returns:
            Dictionary with gate distance parameters
        """
        return {
            "approach_dist": self.get("trajectory_planner.approach_dist", [0.2, 0.3, 0.2, 0.1]),
            "exit_dist": self.get("trajectory_planner.exit_dist", [0.4, 0.15, 0.2, 5.0]),
            "default_approach_dist": self.get("trajectory_planner.default_approach_dist", 0.1),
            "default_exit_dist": self.get("trajectory_planner.default_exit_dist", 0.5),
        }

    def get_height_offsets(self) -> Dict[str, Any]:
        """Get height offset parameters.

        Returns:
            Dictionary with height offset parameters
        """
        return {
            "approach_height_offset": self.get(
                "trajectory_planner.approach_height_offset", [0.01, 0.1, 0.0, 0.0]
            ),
            "exit_height_offset": self.get(
                "trajectory_planner.exit_height_offset", [0.1, 0.0, 0.05, 0.0]
            ),
            "default_approach_height_offset": self.get(
                "trajectory_planner.default_approach_height_offset", 0.1
            ),
            "default_exit_height_offset": self.get(
                "trajectory_planner.default_exit_height_offset", 0.0
            ),
        }

    def get_speed_params(self) -> Dict[str, float]:
        """Get speed parameters for trajectory planning.

        Returns:
            Dictionary with speed parameters
        """
        return {
            "base_speed": self.get("trajectory_planner.speeds.base_speed", 1.8),
            "high_speed": self.get("trajectory_planner.speeds.high_speed", 2.2),
            "approach_speed": self.get("trajectory_planner.speeds.approach_speed", 1.5),
            "exit_speed": self.get("trajectory_planner.speeds.exit_speed", 2.2),
        }

    def get_optimization_params(self) -> Dict[str, float]:
        """Get optimization parameters.

        Returns:
            Dictionary with optimization parameters
        """
        return {
            "drone_clearance_horizontal": self.get(
                "trajectory_planner.optimization.drone_clearance_horizontal", 0.2
            ),
            "gate_half_width": self.get("trajectory_planner.optimization.gate_half_width", 0.25),
            "position_diff_threshold": self.get(
                "trajectory_planner.optimization.position_diff_threshold", 0.05
            ),
            "replanning_threshold": self.get(
                "trajectory_planner.optimization.replanning_threshold", 0.02
            ),
        }

    def get_tuning_params(self, mode: str = "balanced") -> Dict[str, float]:
        """Get tuning parameters for different flight modes.

        Args:
            mode: Tuning mode ("conservative", "aggressive", "balanced")

        Returns:
            Dictionary with tuning parameters
        """
        if mode == "conservative":
            return {
                "Q_pos": self.get("tuning.conservative_Q_pos", 10),
                "Q_vel": self.get("tuning.conservative_Q_vel", 0.05),
                "R": self.get("tuning.conservative_R", 0.005),
            }
        elif mode == "aggressive":
            return {
                "Q_pos": self.get("tuning.aggressive_Q_pos", 5),
                "Q_vel": self.get("tuning.aggressive_Q_vel", 0.001),
                "R": self.get("tuning.aggressive_R", 0.001),
            }
        else:  # balanced
            return {
                "Q_pos": self.get("tuning.balanced_Q_pos", 8),
                "Q_vel": self.get("tuning.balanced_Q_vel", 0.01),
                "R": self.get("tuning.balanced_R", 0.01),
            }

    def reload(self) -> None:
        """Reload constants from the configuration file."""
        self._constants = self._load_constants()

    def update_constant(self, key_path: str, value: Any) -> None:
        """Update a constant value at runtime.

        Args:
            key_path: Path to the constant using dot notation
            value: New value for the constant
        """
        keys = key_path.split(".")
        current = self._constants

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def save_constants(self, output_file: str = None) -> None:
        """Save current constants to a file.

        Args:
            output_file: Output file path (default: same as input file)
        """
        if output_file is None:
            output_file = self.config_path

        with open(output_file, "w") as f:
            toml.dump(self._constants, f)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to constants."""
        return self._constants[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in constants."""
        return key in self._constants


# Create a global instance for easy access
controller_config = ControllerConfig()


# Convenience functions for common access patterns
def get_mpc_weights() -> Dict[str, float]:
    """Get MPC weight constants."""
    return controller_config.get_mpc_weights()


def get_replanning_weights() -> Dict[str, float]:
    """Get replanning weight constants."""
    return controller_config.get_replanning_weights()


def get_collision_params() -> Dict[str, Any]:
    """Get collision avoidance parameters."""
    return controller_config.get_collision_avoidance_params()


def get_trajectory_params() -> Dict[str, Any]:
    """Get trajectory planner parameters."""
    return controller_config.get_trajectory_planner_params()


def get_gate_distances() -> Dict[str, Any]:
    """Get gate distance parameters."""
    return controller_config.get_gate_distances()


def get_height_offsets() -> Dict[str, Any]:
    """Get height offset parameters."""
    return controller_config.get_height_offsets()


def get_speed_params() -> Dict[str, float]:
    """Get speed parameters."""
    return controller_config.get_speed_params()


def get_optimization_params() -> Dict[str, float]:
    """Get optimization parameters."""
    return controller_config.get_optimization_params()


def get_tuning_params(mode: str = "balanced") -> Dict[str, float]:
    """Get tuning parameters for different flight modes."""
    return controller_config.get_tuning_params(mode)


def get_constant(key_path: str, default: Any = None) -> Any:
    """Get a constant value using dot notation."""
    return controller_config.get(key_path, default)


def update_constant(key_path: str, value: Any) -> None:
    """Update a constant value at runtime."""
    controller_config.update_constant(key_path, value)


def reload_constants() -> None:
    """Reload constants from the configuration file."""
    controller_config.reload()
